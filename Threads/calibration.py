import datetime
import os
import pathlib
import time
import numpy as np
from PyQt6 import QtCore

class CameraCalibrationThread(QtCore.QThread):
    '''
    Function for calibrating the camera.
    '''
    take_images = QtCore.pyqtSignal()
    finished = QtCore.pyqtSignal()
    progress = QtCore.pyqtSignal(list)
    voltage = QtCore.pyqtSignal(str, str)

    def __init__(self, parent=None) -> None:
        QtCore.QThread.__init__(self, parent)
        self.running            : bool = True
        '''Is the calibration currently running?'''
        self.static             : bool = parent._cal_static.isChecked()
        '''Is the trim value set to a static value?'''
        self.trim_value         : int = parent._cal_trim.value()
        '''What trim value is set?'''
        self.initial            : int = parent._cal_vthp.value()
        '''What is the initial VThP value set to?'''
        self.end                : int = parent._cal_vthp_stop.value()
        '''What is the end VThP value set to?'''
        self.inc                : int = parent._cal_inc.value()
        '''What is the step size for VThP value set to?'''
        self.vThN               : int = parent._cal_vthn.value()
        '''What is VThN value set to?'''
        self.current            : int = 0
        '''What step is the process currently on?'''
        self.directory          : str = parent._file_dir_2.text()
        '''Where is the data being saved to?'''
        self.queue = parent.cal_queue
        '''Queue used to store calibration arrays'''
        self.pymms = parent.pymms
        '''Class used to communicate with the camera'''

    def cls(self) -> None:
        '''This function clears the console as it can causes memory issues when calibrating.'''
        os.system('cls' if os.name=='nt' else 'clear')

    def create_filename(self) -> None:
        '''
        As the calibration runs files are created to store the data.\n
        Ensure the user specified file is valid and dont overwrite existing files.
        '''
        # The user may try writing to somewhere they shouldn't
        if os.access(self.directory, os.W_OK) is not True:
            print('Invalid directory, writing to Documents')
            self.directory = str(pathlib.Path.home() / 'Documents')

        #Create a filename for saving data
        filename = f'{os.path.join(self.directory)}/P{self.current}_N{self.vThN}'
        if self.static: filename += '_static'
        filename += '_0000.csv'
        #Check if file already exists so you dont overwrite data
        fid = 1
        while os.path.exists(filename):
            filename = f'{filename[:-9]}_{fid:04d}.csv'
            fid+=1
        self.filename = filename
    
    def run(self) -> None:
        # Calculate the number of steps the process needs to run
        number_of_runs = int(((self.end - self.initial) / self.inc) * 81)
        # Scan values 14 through 4, 15 used to find mean, 0,1,2,3 are too intense to use
        for v in range(self.trim_value, 3, -1):
            # Setup the counters for determining how far along the calibration is
            step_counter = 0
            current_percent = 0
            start = time.time()
            # Scan from lower to upper VThP values
            for vthp in range(self.initial, self.end+self.inc, self.inc):
                # Check if user has stopped process
                if not self.running: break
                # Check if user is running a static scan
                if self.static and self.trim_value != v: break
                self.current = vthp
                self.create_filename()
                # Before calibration set VthP and VthN
                self.pymms.calibrate_pimms(update=True,vThN=self.vThN,vThP=vthp)
                # Update the current voltage and trim labels
                self.voltage.emit(f'{vthp}', f'{v}')
                calibration = np.zeros((4,324,324), dtype=np.uint16) # Calibration Array
                # We need to scan 324*324 pixels in steps of 9 pixels, thus 81 steps
                for i in range(0, 81):
                    if not self.running: break
                    self.pymms.calibrate_pimms(value=v,iteration=i)
                    QtCore.QThread.msleep(10)
                    if self.running: self.take_images.emit()
                    # Wait for acquisition to finish
                    array = np.zeros((4,324,324), dtype=np.uint16) # Empty Calibration Array
                    while self.running:
                        # Wait for data to come through queue
                        if self.queue.empty(): 
                            QtCore.QThread.msleep(1)
                            continue
                        # When data comes through queue get it and proceed
                        array = self.queue.get_nowait()
                        break
                    calibration = np.add(calibration,array)
                    # Update the progress bar for how far along the process is
                    percent_complete = int(np.floor((step_counter/number_of_runs) * 100))
                    if percent_complete > current_percent:
                        time_remaining = int(((time.time() - start) / step_counter) * (number_of_runs - step_counter))
                        time_converted = f'{datetime.timedelta(seconds=time_remaining)}'
                        self.progress.emit([time_converted, percent_complete])
                        current_percent = percent_complete
                    step_counter+=1
                    self.cls()
                    print('Done')

                with open(self.filename, "a") as opf:
                    opf.write(f'# Trim Value: {v}\n')
                    np.savetxt(opf, np.sum(calibration,axis=0,dtype=np.int16), delimiter=',', fmt='%i')

                del calibration
            
            # Emit signal to restart counter
            self.progress.emit(['00:00:00', 0])

        # After all threshold values have finished let the UI know the process is done
        if self.running: self.finished.emit()