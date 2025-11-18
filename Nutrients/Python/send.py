import serial
import time

# Set up the serial connection on COM3 with the appropriate baud rate, timeout, etc.
ser = serial.Serial('COM3', baudrate=9600, timeout=1)

# Allow the connection to establish
time.sleep(2)

# Send a command over Bluetooth (replace 'YOUR_COMMAND' with the actual command you want to send)
command = 'YOUR_COMMAND'  # E.g., 'AT' for a simple test command

# Write the command to the Bluetooth device
ser.write(command.encode())

# Read the response (if any)
response = ser.readline()
print(f"Response: {response.decode().strip()}")

# Close the connection after communication is done
ser.close()