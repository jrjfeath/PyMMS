# PyMMS
 Python Controller for PIMMS2 Camera. 
 
 The best way to run this software is through the Run_PymMS.bat as it will remain open if any errors are encountered during runtime so they can be solved.
 
 This viewer should allow PyMMS2 cameras to run at 30fps with 3 bins and no analogue output running. 
 
 Control for the delay stage can be found in the second tab. This tab is optional and a delay stage is not required to run the code. I do not have the hardware ID for delay stages outside of B2 so you may need to figure out which option is the delay stage by attempting to connect to them.
 
# Installation
 For the easiest install extract the files, open command prompt, and change to the directory containing the files and then use the following command:
 pip install -r requirements.txt
 
 Inside requirements.txt you will find the following dependencies:
 h5py
 matplotlib
 pyqt6
 pyqtgraph
 pyyaml
 pyserial
 pythonnet