"""Version-independent, deterministic causal sampling. Standard library only."""
import bisect
import hashlib
import json
import math
from pathlib import Path

from raw_dataset import episode_paths, inspect_episode, jsonl
from raw_episode_writer import CAMERAS

DEFAULT_PROFILE = {'fps':30,'image_max_age_ms':50,'feedback_max_age_ms':50,
    'gripper_max_age_ms':250,'telemetry_max_age_ms':50,'inactive_arm':'hold_observed',
    'gripper_without_target':'hold_observed_if_uncommanded',
    'release_action':'observed_hold_after_acknowledged_stop','grid_origin':'start_plus_one_period',
    'include_preview':False, 'strict_whole_episode':True}


def finite_vector(value,length):
    return isinstance(value,(tuple,list)) and len(value)==length and all(
        type(x) in (int,float) and math.isfinite(x) for x in value)


class Timeline:
    def __init__(self,rows,key='receive_monotonic_ns'):
        value = key if callable(key) else lambda row: row[key]
        self.rows = sorted(rows,key=value)
        self.times = [value(row) for row in self.rows]

    def at(self,stamp):
        index = bisect.bisect_right(self.times,stamp)-1
        return self.rows[index] if index>=0 else None


def source_age(stamp,source,max_ms,label):
    if type(source) is not int or not 0<=stamp-source<=max_ms*1e6:
        raise ValueError(label+':missing_or_stale')
    return (stamp-source)/1e6


def arm_values(row,stamp,profile,side,feedback_row=None,gripper_row=None,accepted_row=None):
    if row is None:
        raise ValueError(side+':missing_telemetry')
    source_age(stamp,row['receive_monotonic_ns'],profile['telemetry_max_age_ms'],side+':telemetry')
    observed=feedback_row if feedback_row is not None else row
    feedback_ns=observed.get('robot_feedback_monotonic_ns')
    feedback_age=source_age(stamp,feedback_ns,profile['feedback_max_age_ms'],side+':feedback')
    joints=observed.get('robot_joints_deg')
    if not finite_vector(joints,6):
        raise ValueError(side+':missing_joint_feedback')
    gripper=row.get('gripper') or {}
    actual=((gripper_row if gripper_row is not None else row).get('gripper') or {}).get('feedback') or {}
    opening=actual.get('opening_controller_units')
    gripper_age=source_age(stamp,actual.get('receive_monotonic_ns'),profile['gripper_max_age_ms'],side+':gripper')
    if not actual.get('valid') or type(opening) not in (int,float) or not math.isfinite(opening):
        raise ValueError(side+':invalid_gripper_feedback')
    command=row.get('last_actually_sent_command') or {}
    active=row.get('clutch_active')
    streaming=row.get('motion_streaming_active')
    stopped=row.get('stop_confirmed_monotonic_ns')
    command_stamp=command.get('send_monotonic_ns')
    stop_acknowledged=(type(stopped) is int and stopped<=stamp and
                       (command_stamp is None or type(command_stamp) is int and stopped>=command_stamp))
    if active and streaming and finite_vector(command.get('joints_deg'),6) and type(command.get('send_monotonic_ns')) is int and command['send_monotonic_ns']<=stamp:
        action=command['joints_deg']
        action_source='actually_sent'
    elif active is False and streaming is False and observed.get('robot_feedback_stationary') is True and profile['inactive_arm']=='hold_observed':
        action=joints
        action_source='derived_hold'
    elif (profile.get('release_action')=='observed_hold_after_acknowledged_stop' and
          streaming is False and profile['inactive_arm']=='hold_observed' and
          ((active is True and row.get('state')=='ANCHORED' and
            observed.get('robot_feedback_stationary') is True) or
           (active is False and row.get('state') in ('RELEASED','INPUT_GATE_CLOSED') and stop_acknowledged))):
        # A position-action encoding of acknowledged stop/initial anchoring.
        # This is explicitly DERIVED from real feedback, including deceleration;
        # it is not a sent command, a claim of physical rest or a cancelled goal.
        action=joints
        action_source='derived_anchor_hold' if active else 'derived_stop_hold'
    else:
        raise ValueError(side+':stop_unconfirmed_or_action_ambiguous')
    accepted=((accepted_row if accepted_row is not None else row).get('gripper') or {}).get('last_accepted_command') or {}
    target=accepted.get('target_position_controller_units')
    ack=accepted.get('ack_monotonic_ns')
    gripper_action_source='controller_accepted'
    if (accepted.get('outcome')!='CONTROLLER_ACCEPTED' or type(ack) is not int or ack>stamp
            or type(target) not in (int,float) or not math.isfinite(target)):
        if (profile['gripper_without_target']=='hold_observed_if_uncommanded' and
                not accepted and gripper.get('enabled') is True and gripper.get('status')=='ONLINE' and
                not gripper.get('fault') and gripper.get('last_attempt') is None and
                gripper.get('pending_action') is None and gripper.get('accepted_command_count')==0):
            target=opening
            ack=None
            gripper_action_source='derived_uncommanded_hold'
        else:
            raise ValueError(side+':no_causal_accepted_gripper_target')
    trace={'telemetry_sequence':row['source_sequence'],'control_tick':row.get('control_tick'),
        'feedback_sequence':observed.get('robot_feedback_sequence'),'feedback_ns':feedback_ns,'feedback_age_ms':feedback_age,
        'feedback_telemetry_sequence':observed.get('source_sequence'),
        'command_sequence':command.get('sequence') if action_source=='actually_sent' else None,
        'command_send_ns':command.get('send_monotonic_ns') if action_source=='actually_sent' else None,
        'action_source':action_source,'clutch_active':active,'state':row.get('state'),
        'stop_acknowledged_ns':stopped if stop_acknowledged else None,
        'feedback_stationary':observed.get('robot_feedback_stationary'),
        'gripper_response_ns':actual['receive_monotonic_ns'],'gripper_age_ms':gripper_age,
        'gripper_feedback_telemetry_sequence':(gripper_row or row).get('source_sequence'),
        'gripper_target_ack_ns':ack,'gripper_action_source':gripper_action_source,
        'gripper_response_is_device_sampling_time':False}
    return [math.radians(x) for x in joints]+[opening/1000], [math.radians(x) for x in action]+[target/1000],trace


class ArmTimeline:
    """Control, joint feedback and gripper responses have independent clocks.

    A feedback response may be bundled into a later telemetry publication. Its
    own receive timestamp remains the causal boundary; future feedback and
    future command acknowledgements are never used.
    """
    def __init__(self, rows):
        self.control=Timeline(rows)
        self.feedback=Timeline([row for row in rows if type(row.get('robot_feedback_monotonic_ns')) is int],
                               key=lambda row: row['robot_feedback_monotonic_ns'])
        feedback_time=lambda row: ((row.get('gripper') or {}).get('feedback') or {}).get('receive_monotonic_ns')
        self.gripper=Timeline([row for row in rows if type(feedback_time(row)) is int], key=feedback_time)
        ack_time=lambda row: ((row.get('gripper') or {}).get('last_accepted_command') or {}).get('ack_monotonic_ns')
        self.accepted=Timeline([row for row in rows if type(ack_time(row)) is int], key=ack_time)

    def values(self, stamp, profile, side):
        return arm_values(self.control.at(stamp),stamp,profile,side,
                          self.feedback.at(stamp),self.gripper.at(stamp),self.accepted.at(stamp))


def plan_episode(path,profile):
    report=inspect_episode(path)
    info=report.get('episode',{})
    reasons=list(report['errors'])
    if not report.get('integrity',{}).get('complete'):
        reasons.append('raw_incomplete')
    if info.get('deleted') or not info.get('retained'):
        reasons.append('deleted_or_not_retained')
    if info.get('mode')!='EXECUTE' and not profile['include_preview']:
        reasons.append('preview_excluded')
    if info.get('termination')!='A_END':
        reasons.append('interrupted_or_failed')
    if not isinstance(info.get('task'),str) or not info['task'].strip():
        reasons.append('missing_task')
    result={'path':str(path),'episode_id':path.name,'accepted':False,'reasons':reasons,'samples':[],
            'derived_hold_count':0,'quality_failures':[],'task':info.get('task')}
    if reasons:
        return result
    initial=json.loads((path/'initial_state.json').read_text())['sides']
    sides={side:ArmTimeline(([initial[side]] if initial.get(side) else [])+jsonl(path/(side+'.jsonl')))
           for side in ('left','right')}
    cameras={c:Timeline(jsonl(path/'frames'/(c+'.jsonl'))) for c in CAMERAS}
    start,end=info['start_ns'],info['end_ns']
    # First grid point is one period after A. This is a fixed documented offset,
    # not silent trimming to the first usable sample. Subsequent gaps reject the episode.
    index=1
    while True:
        stamp=start+(index*1_000_000_000)//profile['fps']
        if stamp>=end:
            break
        try:
            state,action,trace,images=[],[],{},{}
            for side in ('left','right'):
                observed,target,provenance=sides[side].values(stamp,profile,side)
                state.extend(observed);action.extend(target);trace[side]=provenance
            for camera in CAMERAS:
                frame=cameras[camera].at(stamp)
                if not frame:
                    raise ValueError(camera+':missing_image')
                age=source_age(stamp,frame['receive_monotonic_ns'],profile['image_max_age_ms'],camera)
                images[camera]=frame['image_path']
                trace[camera]={'source_sequence':frame['source_sequence'],'frame_number':frame['frame_number'],
                    'receive_monotonic_ns':frame['receive_monotonic_ns'],'age_ms':age,'sha256':frame['sha256']}
            result['samples'].append({'grid_index':index-1,'raw_monotonic_ns':stamp,
                'state':state,'action':action,'images':images,'sources':trace})
            result['derived_hold_count']+=sum(trace[side]['action_source'].startswith('derived_') for side in ('left','right'))
        except ValueError as exc:
            result['quality_failures'].append({'grid_index':index-1,'raw_monotonic_ns':stamp,'reason':str(exc)})
        index+=1
    result['grid_count']=index-1
    if not result['grid_count']:
        result['reasons'].append('zero_training_frames')
    if result['quality_failures']:
        result['reasons'].append('quality_failed_on_fixed_grid')
    result['accepted']=not result['reasons']
    return result


def snapshot_source(path):
    # Include all index/config/state bytes and independently hashed JPEG entries.
    digest=hashlib.sha256()
    for p in sorted(path.rglob('*')):
        if p.is_file() and p.suffix in ('.json','.jsonl'):
            digest.update(str(p.relative_to(path)).encode());digest.update(p.read_bytes())
    return digest.hexdigest()


def make_plan(root,profile=None):
    profile=dict(DEFAULT_PROFILE,**(profile or {}))
    if type(profile['fps']) is not int or not 1<=profile['fps']<=120:
        raise ValueError('fps must be integer 1..120')
    for key in ('image_max_age_ms','feedback_max_age_ms','gripper_max_age_ms','telemetry_max_age_ms'):
        if type(profile[key]) not in (int,float) or not math.isfinite(profile[key]) or profile[key]<=0:
            raise ValueError('invalid profile '+key)
    if (profile['inactive_arm']!='hold_observed' or
            profile['gripper_without_target'] not in ('reject','hold_observed_if_uncommanded') or
            profile.get('release_action') not in ('reject_until_stationary','observed_hold_after_acknowledged_stop') or
            not profile['strict_whole_episode']):
        raise ValueError('unsupported action or segmentation policy')
    plans=[]
    # Snapshot the closed/retained directory list once. Active/finalizing episodes
    # are reported as excluded and never opened by a format writer.
    for path in episode_paths(root):
        before=snapshot_source(path)
        try:
            plan=plan_episode(path,profile)
        except (OSError,ValueError,KeyError,TypeError) as exc:
            plan={'episode_id':path.name,'path':str(path),'accepted':False,
                  'reasons':['invalid_raw:'+str(exc)],'samples':[]}
        plan['raw_sha256']=before
        if before!=snapshot_source(path):
            plan['accepted']=False;plan['reasons'].append('source_changed_during_snapshot')
        plans.append(plan)
    return {'schema':'pico_export_plan_v1','source':str(Path(root).resolve()),'profile':profile,'episodes':plans,
            'sampling_revision':'20261020_independent_feedback_and_stop_hold',
            'action_semantics':{
                'actually_sent':'causal successfully sent joint position target',
                'derived_hold':'fresh observed position while released and stationary',
                'derived_anchor_hold':'fresh observed position during stationary Grip anchoring',
                'derived_stop_hold':'fresh observed position after acknowledged stop; physical settling may continue',
                'derived_uncommanded_hold':'fresh observed gripper opening with no attempted or pending command',
                'feedback_selection':'each response selected by its own host receive timestamp, never future feedback'}}
