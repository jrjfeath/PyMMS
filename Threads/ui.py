import numpy as np
from PyQt6 import QtCore

class CameraCommandsThread(QtCore.QThread):
    '''
    Thread controls camera updates so the UI is not locked.
    This allows the UI to update the progress bar for the user.
    '''
    console_message = QtCore.pyqtSignal(str)
    turn_on_completed = QtCore.pyqtSignal()
    run_camera = QtCore.pyqtSignal()
    finished = QtCore.pyqtSignal()

    def __init__(self, parent=None) -> None:
        QtCore.QThread.__init__(self, parent)
        self.output    : int = parent._exp_type.currentIndex()
        '''What is the output type? Experiment And/No Analogue'''
        self.trigger   : int = parent._trigger.currentIndex()
        '''What is the trigger type?'''
        self.rows      : int = parent._bins.value() + parent._exp_type.currentIndex()
        '''How many rows are being output?'''
        self.trim_file : str = parent._trim_dir.text()
        '''Is there a trim file present?'''
        self.function  : str = parent.function
        '''What function is being run?'''
        self.pymms = parent.pymms 
        '''Class for camera communication'''

    def run(self) -> None:
        # The turn_on function has no arguments
        function = getattr(self.pymms, self.function)
        if self.function == 'turn_on_pimms': function()
        elif self.function == 'start_up_pimms': function(self.trim_file,self.output,self.trigger,self.rows)
        else: function(self.output,self.trigger,self.rows)
        self.console_message.emit(self.pymms.idflex.message)
        if self.function == 'turn_on_pimms': self.turn_on_completed.emit()
        else: self.run_camera.emit()
        self.finished.emit()

class ProgressBarThread(QtCore.QThread):
    progressChanged = QtCore.pyqtSignal(int)
    started = QtCore.pyqtSignal()
    finished = QtCore.pyqtSignal()

    def __init__(self,parent=None) -> None:
        QtCore.QThread.__init__(self, parent)
        self.running = True

    def run(self) -> None:
        self.started.emit()
        while self.running == True:
            self.progressChanged.emit(np.random.randint(low=1,high=100))
            QtCore.QThread.msleep(200)
        self.progressChanged.emit(0)
        self.finished.emit()

class UpdatePlotsThread(QtCore.QThread):
    update = QtCore.pyqtSignal()

    def __init__(self, parent=None) -> None:
        QtCore.QThread.__init__(self, parent)
        self.running = True
        self.waiting = True

    def run(self) -> None:
        while self.running == True:
            if self.waiting:
                QtCore.QThread.msleep(1)
                continue
            self.update.emit()
            QtCore.QThread.msleep(100)

    def pause(self) -> None:
        self.waiting = True

    def stop(self) -> None:
        self.running = False