import sys
import clr

#What directory is the dll stored in?
directory = r"Q:\Cameras\PIMMS"
#What is the name of the newport dll?
file_name = 'Newport.DLS.CommandInterfaceDLS'

#Load in newport c++ library
sys.path.append(directory)
clr.AddReference(file_name)
from CommandInterfaceDLS import *

instrument="COM6"
myDLS = DLS() #DLS is imported from CommandInterfaceDLS
result = myDLS.OpenInstrument(instrument)
if result == 0:
    print(f'Open port {instrument} => Successful')
else:
    print(f'Open port {instrument} => failure', result)

#result = myDLS.RS()
#print ('Reset controller', result)

result = myDLS.VE()
print ('version => ', result)

result = myDLS.TP() #Returns currently position
print('Position',result)

result = myDLS.VA_Set(50.0) #Maximum velocity is 300 mm/s
print('Set Velocity',result)

result = myDLS.VA_Get() #Maximum velocity is 300 mm/s
print('Velocity',result)

result = myDLS.AC_Set(3900.0) #Maximum value of 3900 mm/s^2
print('Set Acceleration',result)

result = myDLS.AC_Get() #Maximum value of 3900 mm/s^2
print('Acceleration',result)

result = myDLS.PG_Get()
print('Distance per move',result)

#result = myDLS.PD(10) #Move relative to current position
#print('Moving:',result)

#result = myDLS.PA_Set(10) #Move to position
#print('Moving:',result)

result = myDLS.TP()
print('Position',result)

result = myDLS.SL_Get()
print('Neg',result)

result = myDLS.SR_Get()
print('Pos',result)

result = myDLS.TP()
print('Position',result)

result = myDLS.TP()
print('Position',result)

result = myDLS.TP()
print('Position',result)

myDLS.CloseInstrument()
