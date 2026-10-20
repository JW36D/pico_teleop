"""Historical Grip-segment recorder retained only for regression tests."""
import copy
import json
from pathlib import Path
import queue
import threading
from incremental import MappingError
from control_sample import CARTESIAN_RESIDUAL_ANNOTATION_TOLERANCES

class DataRecorder:
    def __init__(self, directory, metadata, queue_size=4096):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.metadata = dict(metadata)
        self.metadata['schema'] = 'pico_realman_actions_v2'
        self.metadata['feedback_clock'] = {
            'receive_clock':'local_monotonic_ns', 'robot_sample_clock_available':False,
            'semantics':'TCP response receive time; not robot joint acquisition time',
            'deduplicate_by':'robot_feedback_sequence',
            'configured_feedback_hz_is_target_not_measured_rate':True}
        self.metadata.setdefault('camera', {'available':False, 'synchronization':'not_connected'})
        self.metadata['action_fields'] = {
            'intent':'requested_pose_wxyz before execution constraints',
            'candidate':'command_pose_wxyz and command_joints_deg after shaping/IK',
            'sent':'last_actually_sent_command only after successful motion sender call',
            'observed':'robot_joints_deg is unfiltered real feedback, possibly reused across ticks'}
        self.metadata['training_selection'] = 'quality flags are recorded; no automatic smoothing or label substitution'
        self.metadata['quality_flags_are_annotations_not_training_rejection'] = True
        self.metadata['cartesian_residual_annotation_tolerances'] = CARTESIAN_RESIDUAL_ANNOTATION_TOLERANCES.copy()
        self.metadata['recording_complete_semantics'] = 'all submitted samples persisted; not a claim of task success or training eligibility'
        self.queue = queue.Queue(maxsize=queue_size)
        self.error = None
        self.submitted = self.written = 0
        self.closed = False
        self.last_feedback_key = self.last_pico_sequence = None
        self.last_actually_sent_command = None
        self.episode_id = self.segment_id = 0
        self.last_segment_category = None
        self.was_active = False
        self.file = (self.directory/'samples.jsonl').open('w', buffering=65536)
        (self.directory/'meta.json').write_text(json.dumps(self.metadata, indent=2, allow_nan=False)+'\n')
        self.thread = threading.Thread(target=self.run, name='teleop-data-writer', daemon=True)
        self.thread.start()

    def enrich(self, sample):
        sample = copy.deepcopy(sample)
        sample['record_sequence'] = self.submitted+1
        if 'control_tick' not in sample:
            return sample
        quality = sample.setdefault('quality', {})
        feedback_key = sample.get('robot_feedback_sequence')
        if feedback_key is None:
            feedback_key = sample.get('robot_feedback_monotonic_ns')
        quality['feedback_identity_available'] = feedback_key is not None
        quality['feedback_reused'] = feedback_key is not None and feedback_key == self.last_feedback_key
        quality['new_feedback_response'] = feedback_key is not None and not quality['feedback_reused']
        if feedback_key is not None:
            self.last_feedback_key = feedback_key
        pico_sequence = sample.get('pico_sequence')
        quality['pico_frame_reused'] = pico_sequence is not None and pico_sequence == self.last_pico_sequence
        if pico_sequence is not None:
            self.last_pico_sequence = pico_sequence
        sent = sample.get('last_actually_sent_command')
        if sent is not None:
            self.last_actually_sent_command = copy.deepcopy(sent)
        elif self.last_actually_sent_command is not None:
            sample['last_actually_sent_command'] = copy.deepcopy(self.last_actually_sent_command)
        active = bool(sample.get('clutch_active'))
        if active and not self.was_active:
            self.episode_id += 1
        category = ('preview' if sample.get('mode') == 'PREVIEW_NO_MOTION' else
                    'inactive' if not active else
                    'input_interruption' if quality.get('input_gap_or_recovery') else
                    'feedback_interruption' if quality.get('feedback_transport_transient') or
                        quality.get('feedback_transport_recovered') else
                    'boundary_intervention' if quality.get('boundary_intervention') else
                    'nominal')
        category_changed = category != self.last_segment_category
        if category_changed or active != self.was_active or sample.get('state') == 'ANCHORED':
            self.segment_id += 1
        sample.setdefault('episode_id', self.episode_id if active else None)
        sample.setdefault('segment_id', self.segment_id)
        sample['segment_category'] = category
        sample['segment_start'] = category_changed or active != self.was_active or sample.get('state') == 'ANCHORED'
        self.was_active, self.last_segment_category = active, category
        return sample

    def append(self, sample):
        if self.error:
            raise MappingError('Data recording unavailable: '+self.error)
        if self.closed:
            raise MappingError('Data recording is closed')
        try:
            self.queue.put_nowait(self.enrich(sample))
        except queue.Full:
            self.error = 'Data recording queue full; session incomplete, no samples silently dropped'
            raise MappingError(self.error)
        self.submitted += 1

    def run(self):
        try:
            while True:
                item = self.queue.get()
                if item is None:
                    break
                self.file.write(json.dumps(item, separators=(',', ':'), allow_nan=False)+'\n')
                self.written += 1
        except Exception as exc:
            self.error = str(exc)
        finally:
            try:
                self.file.close()
            except Exception as exc:
                self.error = self.error or str(exc)

    def close(self):
        if not self.closed:
            self.closed = True
            if self.thread.is_alive():
                try:
                    self.queue.put(None, timeout=2.)
                except queue.Full:
                    self.error = self.error or 'Writer did not drain queue'
                self.thread.join(timeout=5.)
                if self.thread.is_alive():
                    self.error = self.error or 'Writer did not finish'
            result = {'submitted_samples':self.submitted, 'written_samples':self.written,
                      'complete':not self.error and self.submitted == self.written,
                      'collection_available':not self.error,
                      'episode_count':self.episode_id, 'segment_count':self.segment_id,
                      'error':self.error}
            try:
                (self.directory/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
            except Exception as exc:
                self.error = self.error or 'Cannot write recording integrity summary: '+str(exc)
                raise MappingError('Data recording unavailable: '+self.error)
        return str(self.directory)
