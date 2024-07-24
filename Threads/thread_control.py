import queue
from PyQt6 import QtCore
from .acquisition import ImageAcquisitionThread, RunCameraThread
from .calibration import CameraCalibrationThread
from .delay import GetDelayPositionThread, MovePositionThread
from .ui import ProgressBarThread, CameraCommandsThread, UpdatePlotsThread

class ThreadControl():
    '''
    Class that holds and controls all of the running threads
    '''
    def __init__(self) -> None:
        self.threads = {}

    def generate_thread(self, thread_name : str , thread_data : QtCore.QObject) -> None:
        '''Generate the thread objects after UI is initialised.'''
        # Check if thread is running, if so kill it before starting again
        if thread_name in self.threads: self.stop_thread(thread_name)
        self.threads[thread_name] = thread_data
        self.threads[thread_name].start()

    def start_thread(self, thread_name : str) -> None:
        '''Tell the thread to start collecting data'''
        # Don't try to start a thread that doesn't exist
        if thread_name not in self.threads: return
        self.threads[thread_name].waiting = False

    def stop_thread(self, thread_name : str) -> None:
        '''Tell the thread to stop collecting data'''
        # Don't try to start a thread that doesn't exist
        if thread_name not in self.threads: return
        self.threads[thread_name].pause()

    def kill_thread(self, thread_name : str) -> None:
        '''Safely close thread.'''
        # Don't try to kill a thread that doesn't exist
        if thread_name not in self.threads: return
        self.threads[thread_name].running = False
        self.threads[thread_name].wait()
        self.threads.pop(thread_name, None)

    def start_camera_update(self, parent, function : str):
        '''
        This function controls updating camera settings. \n
        It terminates on its own after settings are updated.
        '''
        parent.function = function
        # Setup a progressbar to indicate camera communication
        thread = ProgressBarThread()
        thread.started.connect(lambda: parent._progressBar.setFormat('Communicating with camera.'))
        thread.progressChanged.connect(parent.update_pb)
        thread.finished.connect(lambda: parent._progressBar.setFormat(''))
        self.generate_thread('ProgressBar', thread)
        # Start the camera communication thread
        thread = CameraCommandsThread(parent)
        thread.console_message.connect(parent.update_console)
        thread.turn_on_completed.connect(parent.lock_camera_connect)
        thread.run_camera.connect(parent.unlock_run_camera)
        thread.finished.connect(self.finished_camera_update)
        self.generate_thread('Camera', thread)

    def finished_camera_update(self) -> None:
        self.kill_thread('ProgressBar')
        self.threads.pop('Camera', None)

    def spawn_acquisition_threads(self,parent) -> None:
        '''Called on startup, to initialise acquisition threads in background'''
        if 'Images' not in self.threads: 
            thread = ImageAcquisitionThread()
            self.generate_thread('Images', thread)
        if 'Acquisition' not in self.threads:
            # Generate shared queue objects for storing image and calibration data
            parent.cal_queue = queue.Queue(maxsize=2)
            parent.image_queue = queue.Queue(maxsize=2)
            thread = RunCameraThread()
            thread.acquisition_fps.connect(parent.update_fps)
            thread.tof_counts.connect(parent.update_tof_counts)
            thread.ui_image.connect(parent.update_image)
            thread.progress.connect(parent.update_console)
            thread.limit.connect(parent.start_and_stop_camera)
            self.generate_thread('Acquisition', thread)
        if 'Plots' not in self.threads:
            thread = UpdatePlotsThread()
            thread.update.connect(parent.update_plots)
            self.generate_thread('Plots', thread)

    def start_acquisition_threads(self, parent) -> None:
        '''Updates and then unpauses the acquisition threads.'''
        # Check if user is connected to the delay stage and saving data
        if parent.dly_connected and parent._save_box.isChecked():
            self.move_to_starting_position()
        #Fast Pymms has to start the acquisition
        if hasattr(parent.pymms.idflex, 'StartAcquisition'):
            parent.pymms.idflex.StartAcquisition()
        self.threads['Images'].update_variables(parent)
        self.threads['Acquisition'].update_variables(parent)
        self.start_thread('Images')
        self.start_thread('Acquisition')
        self.start_thread('Plots')

    def stop_acquisition_threads(self, parent) -> None:
        '''Pause the acquisition threads.'''
        for thread_name in ['Acquisition','Plots','Images']:
            self.stop_thread(thread_name)
        # Empty image queue after threads are paused
        with parent.image_queue.mutex:
            parent.image_queue.queue.clear()
        # Fast Pymms has to close the acquisition
        if hasattr(parent.pymms.idflex, 'StopAcquisition'):
            parent.pymms.idflex.StopAcquisition()

    def start_camera_calibration(self, parent) -> None:
        print('Setting up for calibration.\n\n')
        # Set the start and stop labels
        parent._cal_start_l.setText(f'{parent._cal_vthp.value()}')
        parent._cal_end_l.setText(f'{parent._cal_vthp_stop.value()}')
        # Create the calibration thread
        thread = CameraCalibrationThread(parent)
        thread.progress.connect(parent.update_calibration_progress)
        thread.take_images.connect(parent.pass_main_for_calibration)
        thread.voltage.connect(parent.update_cal_values)
        thread.finished.connect(lambda: parent.start_and_stop_camera('Calibration'))
        self.generate_thread('Calibration', thread)

    def stop_camera_calibration(self) -> None:
        self.kill_thread('Calibration')

    def start_delay_stage(self, parent) -> None:
        '''Starts the thread that updates the delay stage position'''
        if 'Delay' not in self.threads:
            thread = GetDelayPositionThread()
            thread.progressChanged.connect(parent.update_pos)
            self.start_thread('Delay', thread)

    def stop_delay_stage(self) -> None:
        '''Stops the thread that updates the delay stage position'''
        self.kill_thread('Delay')

    def move_delay_stage(self, parent) -> None:
        '''Move the delay stage to a new position.'''
        if 'Move' not in self.threads:
            thread = MovePositionThread(parent)
            thread.progressChanged.connect(parent.update_pos)
            thread.message.connect(parent.update_console)
            self.start_thread('Move', thread)

    def kill_threads(self) -> None:
        '''Kill all threads before closing the application'''
        for thread_name in list(self.threads):
            self.kill_thread(thread_name)