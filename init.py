import subprocess
import keyboard
import os

def main():
    
    #initiate frame_capture and lowlight scripts

    frame_capture_process = subprocess.Popen(['python', 'frame_capture.py'])

    lowlight_process = subprocess.Popen(['python', os.path.join('Zero-DCE', 'lowlight_test.py')])

    print("Press 'Esc' to stop")
    try:
        # Listen for Escape keypress event
        keyboard.wait('esc')
    except KeyboardInterrupt:
        pass  

    frame_capture_process.terminate()
    lowlight_process.terminate()

if __name__ == '__main__':
    main()
