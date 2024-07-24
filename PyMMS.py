#####################################################################################
# Variables found throughout this file are defined as:
# 1) UI variables from image.ui are: _name
# 2) UI variables generated in this file are: name_
# 3) All other variables are lowercase: name
#####################################################################################

# Native python library imports
import os
import pathlib
import sys
# External python library imports
import numpy as np
import pyqtgraph as pg
import warnings
import yaml
from PyQt6 import uic, QtWidgets, QtCore, QtGui
from Classes.delay_stage import NewportDelayStage
from Threads.thread_control import ThreadControl
#Supress numpy divide and cast warnings
warnings.simplefilter('ignore', RuntimeWarning)

#########################################################################################
# Class used for modifying the plots
#########################################################################################
class UI_Plots():
    def __init__(self, main_window):
        self.window_ = main_window
        self.generate_plots()
    
    def generate_plots(self):
        '''
        Setup plot for TOF data \n
        Plot only updates the data and does not redraw each update.\n
        Axes are updated by looking at value changes.\n
        '''
        # Setup plot for image readout
        self.window_.image_widget_origpos = self.window_._image_widget.pos()
        self.window_.image_widget_origwidth = self.window_._image_widget.width()
        self.window_.image_widget_origheight = self.window_._image_widget.height()
        self.window_._image_widget.setWindowTitle('Readout')
        self.window_._image_widget.installEventFilter(self.window_)
        self.grid = QtWidgets.QGridLayout(self.window_._image_widget)
        self.window_.graphics_view_ = pg.RawImageWidget(self.window_._image_widget,scaled=True)
        self.grid.addWidget(self.window_.graphics_view_, 0, 0, 1, 1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.window_.graphics_view_.sizePolicy().hasHeightForWidth())
        self.window_.graphics_view_.setSizePolicy(sizePolicy)
        self.window_.graphics_view_.setImage(self.window_.image_, levels=[0,255])

        self.window_.tof_plot_origpos = self.window_._tof_plot.pos()
        self.window_.tof_plot_origwidth = self.window_._tof_plot.width()
        self.window_.tof_plot_origheight = self.window_._tof_plot.height()
        self.window_.tof_plot_line = self.window_._tof_plot.plot(self.window_.ion_counts_,pen=pg.mkPen(color=(255, 0, 0)))
        self.window_.tof_selection_line_min = self.window_._tof_plot.plot(self.window_.ion_counts_,pen=pg.mkPen(color=(0, 0, 0)))
        self.window_.tof_selection_line_max = self.window_._tof_plot.plot(self.window_.ion_counts_,pen=pg.mkPen(color=(0, 0, 0)))
        self.window_._tof_plot.setWindowTitle('ToF Spectrum')
        self.window_._tof_plot.setBackground('w')
        self.window_._tof_plot.installEventFilter(self.window_)
        self.window_._tof_plot.setLabel('bottom', "Time Bin")
        self.window_._tof_plot.setLabel('left', "Counts")
        
        self.window_.ion_count_plot_origpos = self.window_._ion_count_plot.pos()
        self.window_.ion_count_plot_origwidth = self.window_._ion_count_plot.width()
        self.window_.ion_count_plot_origheight = self.window_._ion_count_plot.height()
        self.window_.ion_count_plot_line = self.window_._ion_count_plot.plot(self.window_.ion_counts_,pen=pg.mkPen(color=(255, 0, 0)))
        self.window_._ion_count_plot.setWindowTitle('Ion Count Spectrum')
        self.window_._ion_count_plot.setBackground('w')
        self.window_._ion_count_plot.installEventFilter(self.window_)
        self.window_._ion_count_plot.setLabel('bottom', "Number of Frames")
        self.window_._ion_count_plot.setLabel('left', "Total Ion Count")

        self.change_axes()

    def plot_selection(self):
        '''Updates the selection of the ToF displayed on the plot.'''
        x = np.zeros(4096)
        x[self.window_._min_x.value()] = self.window_._max_y.value()
        x[self.window_._max_x.value()] = self.window_._max_y.value()
        self.window_.tof_selection_line_min.setData(x,pen=pg.mkPen(color=(0, 0, 0), style=QtCore.Qt.PenStyle.DotLine))
        self.window_.tof_selection_line_max.setData(x,pen=pg.mkPen(color=(0, 0, 0), style=QtCore.Qt.PenStyle.DotLine))

    def change_axes(self):
        '''
        Updates the TOF axes to make it easier to view data.
        '''
        self.window_._tof_plot.setXRange(self.window_._tof_range_min.value(), self.window_._tof_range_max.value(), padding=0)
        self.window_._tof_plot.setYRange(self.window_._min_y.value(), self.window_._max_y.value(), padding=0)
        self.plot_selection()

    def rotate_camera(self,option):       
        #Rotate clockwise
        if option == 0:
            self.window_.rotation_ += 1
            if self.window_.rotation_ == 4: self.window_.rotation_ = 0
        #Rotate counter clockwise
        else:
            self.window_.rotation_ -= 1
            if self.window_.rotation_ < 0: self.window_.rotation_ = 3

    def pop_out_window(self,option):
        if option == 0: #If user is popping out tof
            self.window_.tof_expanded_ = True
            self.window_._tof_plot.setParent(None)
            self.window_._tof_plot.move(int(self.window_.width()/2),int(self.window_.height()/2))
            self.window_._tof_plot.show()
        elif option == 1: #if user is popping out image
            self.window_.img_expanded_ = True
            self.window_._image_widget.setParent(None)
            self.window_._image_widget.move(int(self.window_.width()/2),int(self.window_.height()/2))
            self.window_._image_widget.show()
        else:
            self.window_.ionc_expanded_ = True
            self.window_._ion_count_plot.setParent(None)
            self.window_._ion_count_plot.move(int(self.window_.width()/2),int(self.window_.height()/2))
            self.window_._ion_count_plot.show()             

#########################################################################################
# Mainwindow used for displaying UI
#########################################################################################
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super(MainWindow, self).__init__()

        # Directory containing this file
        fd = os.path.dirname(__file__)

        # Before we proceed check if we can load Parameter file
        defaults_fp = os.path.join(fd,"PyMMS_Defaults.yaml")
        if not os.path.isfile(defaults_fp):
            print("Cannot find the Parameters file (PyMMS_Defaults.yaml), where did it go?")
            exit()
        # Load defaults
        with open(defaults_fp, 'r') as stream:
            self.defaults = yaml.load(stream,Loader=yaml.SafeLoader)

        # Create path to ui file
        uifp = os.path.join(fd,"image.ui")
        if not os.path.isfile(uifp):
            print("Cannot find the ui file (image.ui), where did it go?")
            exit()
        # Load UIC file
        uic.loadUi(uifp,self)

        # Call the class for your respective delay stage
        self.delay_stage = NewportDelayStage(fd)
        # Generate thread class
        self.thread_control = ThreadControl()
        self.thread_control.spawn_acquisition_threads(self)

        #Add default save locations
        self._file_dir_2.setText(str(pathlib.Path.home() / 'Documents'))

        #Add colourmaps
        colourmaps = pg.colormap.listMaps("matplotlib")
        self._colourmap.addItems(colourmaps)

        #Make some UI variables
        self.pymms = None
        self.rotation_ = 0 # Rotation angle of the image
        self.connected_ = False #Is camera connected?
        self.dly_connected = False #Is the delay stage connected?
        self.camera_running_ = False #Is camera running?
        self.run_calibration_ = False
        self.calibration_array_ = np.zeros((4,324,324), dtype=np.uint16)
        self.tof_expanded_ = False
        self.img_expanded_ = False
        self.ionc_expanded_ = False
        self.reset_cml_ = False
        self.error_ = False #Is there an error?
        self.ion_count_displayed = 0
        self.ion_counts_ = [np.nan for _ in range(self._nofs.value())]
        self.tof_counts_ = np.zeros(4096)
        self.image_ = np.zeros((324,324))
        self._vthp.setValue(self.defaults['dac_settings']['vThP'])
        self._vthn.setValue(self.defaults['dac_settings']['vThN'])
        self._bins.setValue(self.defaults['ControlSettings']['Mem Reg Read'][0])

        #Call the class responsible for plot drawing and functions
        self.ui_plots = UI_Plots(self)

        '''
        Various commands for buttons found on the ui
        '''
        self._camera_connect_button.clicked.connect(self.camera_control)
        self._trim_button.clicked.connect(lambda: self.open_file_dialog(1))
        self._path_button.clicked.connect(lambda: self.open_file_dialog(0))
        self._rotate_c.clicked.connect(lambda: self.ui_plots.rotate_camera(0))
        self._rotate_cc.clicked.connect(lambda: self.ui_plots.rotate_camera(1))
        self._update_camera.clicked.connect(self.update_camera)
        self._update_camera_2.clicked.connect(self.update_output)
        self._vthp.valueChanged.connect(self.update_dafaults)
        self._vthn.valueChanged.connect(self.update_dafaults)
        self._bins.valueChanged.connect(self.update_dafaults)
        self._window_view.currentIndexChanged.connect(self.change_image)
        self._view.currentIndexChanged.connect(self.individual_or_cumulative)
        self._reset_cumulative.clicked.connect(self.reset_cumulative)

        self._button.clicked.connect(lambda: self.start_and_stop_camera('Acquisition'))
        self._cal_run.clicked.connect(lambda: self.start_and_stop_camera('Calibration'))

        #Update the plots when they are clicked on
        self._nofs.valueChanged.connect(self.reset_nofs)
        self._tof_range_min.valueChanged.connect(self.ui_plots.change_axes)
        self._tof_range_max.valueChanged.connect(self.ui_plots.change_axes)
        self._min_y.valueChanged.connect(self.ui_plots.change_axes)
        self._max_y.valueChanged.connect(self.ui_plots.change_axes)
        self._min_x.valueChanged.connect(self.ui_plots.plot_selection)
        self._max_x.valueChanged.connect(self.ui_plots.plot_selection)
        self._pop_tof.clicked.connect(lambda: self.ui_plots.pop_out_window(0))
        self._pop_ion_count.clicked.connect(lambda: self.ui_plots.pop_out_window(2))
        self._pop_image.clicked.connect(lambda: self.ui_plots.pop_out_window(1))

        #Sometimes while idle the code tries to quit, prevent closing
        quit = QtGui.QAction("Quit", self)
        quit.triggered.connect(self.closeEvent)

        #Delay stage UI interactions
        self.update_coms()
        self._dly_refresh.clicked.connect(self.update_coms)
        self._dly_connect_button.clicked.connect(self.dly_control)
        self._dly_send_to.clicked.connect(self.dly_position)
        self._dly_vel_but.clicked.connect(self.dly_velocity)
        if not self.delay_stage.dls_files_present:
            self._dly_connect_button.setEnabled(False)

    def update_tof_counts(self, ion_counts : int, tof_counts : np.ndarray) -> None:
        '''Updates the total ion count and the tof spectrum'''
        self.tof_counts_ = tof_counts
        self.ion_count_displayed = ion_counts
        self.ion_counts_.append(ion_counts)
        del self.ion_counts_[0]

    def update_image(self, value : np.ndarray) -> None:
        self.image_ = value

    def update_plots(self) -> None:
        image = self.image_
        
        # Scale the image based off the slider
        if self._vmax.value() != 100:
            image[image > (self._vmax.value()*0.01)] = 0
            image = ((image / np.max(image)))

        colourmap = self._colourmap.currentText()
        if colourmap != "None": 
            cm = pg.colormap.get(colourmap,source="matplotlib")
            image = cm.map(image)
        else:
            image = np.array(image * 255, dtype=np.uint8)

        self.ion_count_plot_line.setData(self.ion_counts_)
        self.tof_plot_line.setData(self.tof_counts_)
        self.graphics_view_.setImage(image, levels=[0,255])
        self._ion_count.setText(str(self.ion_count_displayed))

    def reset_nofs(self) -> None:
        self.ion_counts_ = [np.nan for _ in range(self._nofs.value())]
        self.ion_count_plot_line.setData(self.ion_counts_)

    def reset_plots(self) -> None:
        self.ion_counts_ = [np.nan for _ in range(self._nofs.value())]
        self.tof_counts_ = np.zeros(4096)
        self.image_ = np.zeros((324,324))

    def change_image(self) -> None:
        '''Change which image is being displayed.'''
        self.thread_control.threads['Acquisition'].image_index = self._window_view.currentIndex()

    def individual_or_cumulative(self) -> None:
        '''Does the user want to view the individual or summed images?'''
        self.thread_control.threads['Acquisition'].displayed = self._view.currentIndex()

    def reset_cumulative(self) -> None:
        '''Reset the cumulative images'''
        if not self.thread_control.threads['Acquisition'].waiting:
            self.thread_control.threads['Acquisition'].reset_cumulative()

    def update_pb(self, value : int) -> None:
        '''Set the progressbar value'''
        self._progressBar.setValue(value)

    def update_cal_pb(self, value : int) -> None:
        '''Set the progressbar value'''
        self._cal_progress.setValue(value)

    def update_cal_values(self, vThP_low : str, trim_value : str) -> None:
        self._self_cur_l.setText(vThP_low)
        self._cal_trim_value.setText(trim_value)

    def update_fps(self, frames : str, fps : str) -> None:
        '''Set frame count and the fps for the current acquisition'''
        self._frame_count.setText(frames)
        self._fps_1.setText(fps)

    def update_calibration_progress(self, value : list) -> None:
        self._cal_remaining.setText(value[0])
        self._cal_progress.setValue(value[1])

    def pass_main_for_calibration(self) -> None:
        self.thread_control.start_acquisition_threads(self)

    def open_file_dialog(self,option : int) -> None:
        '''Open the window for finding files/directories'''
        if option == 0:
            dir_path = QtWidgets.QFileDialog.getExistingDirectory(
                parent=self,
                caption="Select directory",
                directory=os.getenv("HOME"),
                options=QtWidgets.QFileDialog.Option.ShowDirsOnly,
            )
            self._dir_name.setText(dir_path)
        else:
            file_path = QtWidgets.QFileDialog.getOpenFileName(
                parent=self,
                caption="Open File",
                directory=os.getenv("HOME")
            )
            self._trim_dir.setText(file_path[0])

    def update_console(self,string : str) -> None:
        '''
        Update console log with result of command passed to PIMMS.
        '''
        if 'Error' in string:
            self.thread_control.kill_thread('ProgressBar')
            self.error_ = True
            self._plainTextEdit.setPlainText(string)
            self._plainTextEdit.setStyleSheet(
                """QPlainTextEdit {color: #FF0000;}"""
            )
            error = QtWidgets.QMessageBox()
            error.setText(string)
            error.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)
            error = error.exec()
        else:
            self._plainTextEdit.setPlainText(string)

    def start_and_stop_camera(self, button=None) -> None:
        '''
        This function starts and stops the camera acquisition.\n
        It disables all buttons except for plot pop-outs and the start/stop.
        '''
        available_buttons = ['_pop_tof','_pop_ion_count','_pop_image',
                             '_rotate_c','_rotate_cc','_reset_cumulative']
        # The button returns none when the frame limit is reached
        if button != None:
            enable = True
            # If the user is doing a simple acquisition
            if button == 'Acquisition':
                available_buttons.append('_button')
                if self.camera_running_:
                    self.thread_control.stop_acquisition_threads(self)
                    self.camera_running_ = False
                    self._button.setText("Start")
                else:
                    self.camera_running_ = True
                    self._button.setText("Stop")
                    self.thread_control.start_acquisition_threads(self)
                    enable = False
            # If the user is looking to run a calibration
            if button == 'Calibration':
                available_buttons.append('_cal_run')
                if self.run_calibration_:
                    self.thread_control.stop_camera_calibration()
                    self._cal_run.setText("Start")
                    self._exp_type.setEnabled(True)
                    self.run_calibration_ = False
                else:
                    self.run_calibration_ = True
                    self._cal_run.setText("Stop")
                    self._bins.setValue(4)
                    self._trigger.setCurrentIndex(0)
                    self._exp_type.setCurrentIndex(0)
                    self._exp_type.setEnabled(False)
                    self.thread_control.start_camera_calibration(self)
                    enable = False
            for button in self.findChildren(QtWidgets.QPushButton):
                if button.objectName() in available_buttons: continue
                button.setEnabled(enable)
        else:
            self.thread_control.stop_acquisition_threads(self)
            # When the user is not running calibration re-enable UI
            if self.camera_running_:
                for button in self.findChildren(QtWidgets.QPushButton):
                    if button.objectName() in available_buttons: continue
                    button.setEnabled(True)
                self._button.setText("Start")
                self.camera_running_ = False

    def eventFilter(self, source, event):
        '''
        Capture close events to pop the windows back into the main body
        '''
        if (event.type() == QtCore.QEvent.Type.Close and isinstance(source, QtWidgets.QWidget)):
            if source.windowTitle() == self._tof_plot.windowTitle():
                self.tof_expanded_ = False
                self._tof_plot.setParent(self._tab)
                self._tof_plot.setGeometry(
                    self.tof_plot_origpos.x(),
                    self.tof_plot_origpos.y(),
                    self.tof_plot_origwidth,
                    self.tof_plot_origheight
                )
                self._tof_plot.show()
            elif source.windowTitle() == self._image_widget.windowTitle():
                self.img_expanded_ = False
                self._image_widget.setParent(self._tab)
                self._image_widget.setGeometry(
                    self.image_widget_origpos.x(),
                    self.image_widget_origpos.y(),
                    self.image_widget_origwidth,
                    self.image_widget_origheight
                )
                self._image_widget.show()
            else:
                self.ionc_expanded_ = False
                self._ion_count_plot.setParent(self._tab)
                self._ion_count_plot.setGeometry(
                    self.ion_count_plot_origpos.x(),
                    self.ion_count_plot_origpos.y(),
                    self.ion_count_plot_origwidth,
                    self.ion_count_plot_origheight
                )
                self._ion_count_plot.show()                          
            event.ignore() #Ignore close
            return True #Tell qt that process done
        else:
            return super(MainWindow,self).eventFilter(source,event)

    def closeEvent(self, event) -> None:
        '''If the application is closed do the following'''
        close = QtWidgets.QMessageBox()
        close.setText("Would you like to quit?")
        close.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Yes | 
                                    QtWidgets.QMessageBox.StandardButton.No)
        close = close.exec()

        if close == QtWidgets.QMessageBox.StandardButton.Yes.value:
            if self.dly_connected:
                self.delay_stage.disconnect_stage()
                pass
            if self.pymms != None: 
                self.thread_control.kill_threads()
            if self.connected_: 
                self.pymms.idflex.close_device()
            self._image_widget.close()
            event.accept()
        else:
            event.ignore()

    #####################################################################################
    # Delay Stage Functions
    #####################################################################################
    def update_pos(self,value : float) -> None:
        '''Updates the position of the delay stage on the UI'''
        self._dly_pos.setText(str(value))

    def update_coms(self) -> None:
        '''Updates the available coms on the computer'''
        self._com_ports.clear()
        com_list = self.delay_stage.get_com_ports()
        self._com_ports.addItems(com_list)

    def dly_control(self) -> None:
        '''Connect to and disconnect from the delay stage.'''
        if self.dly_connected == False:
            port, name, _ = self._com_ports.currentText().split(';')
            if 'Delay Stage' in name:
                self.delay_stage.connect_stage(port)
                self._dly_vel.setText(self.delay_stage.get_velocity())
                self._dly_pos_min.setText(self.delay_stage.get_minimum_position())
                self._dly_pos_max.setText(self.delay_stage.get_maximum_position())
                self.dly_connected = True
                self._dly_connect_button.setText('Disconnect')
                self.thread_control.start_delay_stage(self)
        else:
            self.thread_control.stop_delay_stage()
            self.delay_stage.disconnect_stage()
            self._dly_connect_button.setText('Connect')
            self.dly_connected = False

    def dly_velocity(self) -> None:
        '''Change the velocity of the delay stage'''
        self.delay_stage.set_velocity(self._dly_vel_set.value())
        self._dly_vel.setText(self.delay_stage.get_velocity())

    def dly_position(self) -> None:
        '''Move delay stage to position specified.'''
        self.thread_control.move_delay_stage(self)

    #####################################################################################
    # PIMMS Functions
    #####################################################################################
    def lock_camera_connect(self) -> None:
        '''
        Only connect to the camera and enable sending commands if there are no issues
        '''
        if self.connected_ == False and self.error_ == False:
            self._camera_connect.setText(f'Connected: {self._function_selection.currentText()}')
            self._camera_connect.setStyleSheet("color: green")
            self._camera_connect_button.setText(f'Disconnect')
            self._update_camera.setDisabled(False)
            self.connected_ = True
            self.thread_control.kill_thread('ProgressBar')
        else:
            self.pymms = None
            self._camera_connect.setText(f'Disconnected')
            self._camera_connect.setStyleSheet("color: red")
            self._camera_connect_button.setText(f'Connect')
            self._button.setDisabled(True)
            self._cal_run.setDisabled(True)
            self._update_camera.setDisabled(True)
            self._update_camera_2.setDisabled(True)
            self.connected_ = False
            self.error_ = False
        self._camera_connect_button.setDisabled(False)

    def update_dafaults(self) -> None:
        '''
        Update user controllable default values
        '''
        if self.pymms == None:
            self.defaults['dac_settings']['vThP'] = self._vthp.value()
            self.defaults['dac_settings']['vThN'] = self._vthn.value()
            self.defaults['ControlSettings']['Mem Reg Read'][0] = self._bins.value()
        else:
            self.pymms.settings['dac_settings']['vThP'] = self._vthp.value()
            self.pymms.settings['dac_settings']['vThN'] = self._vthn.value()
            self.pymms.settings['ControlSettings']['Mem Reg Read'][0] = self._bins.value()

    def camera_control(self) -> None:
        '''
        Connect to and disconnect from the camera.
        '''
        if self.connected_ == False:
            # Find which option the user has selected 
            option_index = self._function_selection.currentIndex()
            #Functions to import that relate to PIMMS ideflex.dll
            if option_index == 0:
                from Classes.PyMMS_idflex import pymms
            if option_index == 1:
                from Classes.PyMMS_idflex_fast import pymms
            if option_index == 2:
                from Classes.PyMMS_idflex_test import pymms
            #Create PyMMS object
            self.pymms = pymms(self.defaults)
            # Check if the idflex.dll can be found
            if self.pymms.idflex.error == 1:
                self._plainTextEdit.setPlainText(self.pymms.idflex.message)
                return
            self._plainTextEdit.setPlainText(f'Connecting to Camera, please wait!')
            self._camera_connect_button.setDisabled(True)
            self.thread_control.start_camera_update(self, 'turn_on_pimms')
        else:
            #Disconnect from the camera
            self._plainTextEdit.setPlainText(f'Disconnecting from Camera, please wait!')
            self.pymms.idflex.close_device()
            self._plainTextEdit.setPlainText(self.pymms.idflex.message)
            self.lock_camera_connect()
            self.pymms = None

    def unlock_run_camera(self) -> None:
        self._button.setDisabled(False)
        self._cal_run.setDisabled(False)
        self._update_camera.setDisabled(False)
        self._update_camera_2.setDisabled(False)
        self._camera_connect_button.setDisabled(False)    

    def update_output(self) -> None:
        '''
        If user just wants to update the readout type

        Check that camera has been connected, updated and is not running.
        '''
        self._camera_connect_button.setDisabled(True)
        self._button.setDisabled(True)
        self._cal_run.setDisabled(True)
        self._update_camera.setDisabled(True)
        self._update_camera_2.setDisabled(True)
        self._plainTextEdit.setPlainText(f'Updating Camera output type!')
        self.thread_control.start_camera_update(self, 'send_output_types')

    def update_camera(self) -> None:
        '''
        Update the DAC, camera settings, and TRIM file.
        '''
        if self.connected_ == False: 
            self._plainTextEdit.setPlainText(f'Please connect to the camera before updating output!')
            return
        self._camera_connect_button.setDisabled(True)
        self._button.setDisabled(True)
        self._cal_run.setDisabled(True)
        self._update_camera.setDisabled(True)
        self._update_camera_2.setDisabled(True)
        if os.path.isfile(self._trim_dir.text()) == True:
            self._plainTextEdit.setPlainText(f'Updating Camera DACs with trim file!')
        else:
            self._plainTextEdit.setPlainText(f'Updating Camera DACs without trim file!')
        self.thread_control.start_camera_update(self, 'start_up_pimms')

def except_hook(cls, exception, traceback):
    '''If this is not called the UI will not return errors'''
    sys.__excepthook__(cls, exception, traceback)

if __name__ == '__main__':
    sys.excepthook = except_hook
    app = QtWidgets.QApplication(sys.argv)

    #Create window
    w = MainWindow()
    #Load the app
    w.show()

    app.exec()
