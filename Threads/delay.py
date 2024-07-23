import numpy as np
from PyQt6 import QtCore

class GetDelayPositionThread(QtCore.QThread):
    '''
    Thread reading position data of delay stage for UI
    '''
    progressChanged = QtCore.pyqtSignal(float)
    finished = QtCore.pyqtSignal()

    def __init__(self,parent=None) -> None:
        QtCore.QThread.__init__(self, parent)
        self.running = True
        self.delay_stage = parent.delay_stage

    def run(self) -> None:
        while self.running == True:
            value = self.delay_stage.get_position()
            try: self.progressChanged.emit(float(value))
            except: print('Cannot read position data.')
            QtCore.QThread.msleep(10)
        self.finished.emit()

class MovePositionThread(QtCore.QThread):
    '''
    This function moves the delay stage to the starting position.
    '''
    progressChanged = QtCore.pyqtSignal(int)
    message = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal()

    def __init__(self,parent=None) -> None:
        QtCore.QThread.__init__(self, parent)
        self.running = True
        self.start_pos = parent._delay_t0.value() + (parent._dly_start.value() / 6671.2819)
        self.delay_stage = parent.delay_stage

    def run(self) -> None:
        self.delay_stage.set_position(self.start_pos)
        while self.running == True:
            self.progressChanged.emit(np.random.randint(0,100))
            value = self.delay_stage.get_position()
            try: float(value)
            except: print('Cannot read position data.')
            self.message.emit(f'Moving to start position: {round(self.start_pos,4)}, Current position: {round(value,4)}')
            # Check if position is within bounds
            if (self.start_pos-0.01 < float(value) < self.start_pos+0.01):
                break
            else:
                QtCore.QThread.msleep(10)
        self.progressChanged.emit(0)
        self.message.emit(f'Finished moving to start position')
        self.finished.emit()