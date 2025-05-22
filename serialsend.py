import serial
import time


PORT = 'COM10'      
BAUD_RATE = 9600   

try:

    bt = serial.Serial(PORT, BAUD_RATE, timeout=1)
    time.sleep(2)  

    bt.write(b'd,2\n')

    bt.close()

except serial.SerialException as e:
    print("Failed to connect or send:", e)