import serial
import struct
import time
import sys
import threading
import tty
import termios


PORT = '/dev/serial0'
BAUDRATE = 115200


JUMP_THROTTLE = 1450       
SAFE_HOVER_THROTTLE = 1350  
MAX_THROTTLE = 1700


STEP_THR = 10
STEP_POS = 15 


rc_throttle = 1000
rc_roll = 1500
rc_pitch = 1500
rc_yaw = 1500

is_armed = False
is_surface_mode = False 
running = True

class DroneLink:
    def __init__(self):
        try:
            self.ser = serial.Serial(PORT, BAUDRATE, timeout=1)
        except Exception as e:
            print(f"BŁĄD PORTU: {e}")
            sys.exit(1)

    def set_rc(self, channels):
        payload = struct.pack('<8H', *channels)
        size = len(payload)
        checksum = 0 ^ size ^ 200
        for b in payload: checksum ^= b
        msg = struct.pack('<3c2B', b'$', b'M', b'<', size, 200) + payload + struct.pack('<B', checksum)
        self.ser.write(msg)
        
    def close(self):
        self.ser.close()

def getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def drone_loop(drone):
    global rc_throttle, rc_roll, rc_pitch, rc_yaw, is_armed, is_surface_mode, running
    
    print(">>> SYSTEM GOTOWY. CZEKAM NA KOMENDY.")
    
    while running:

        aux1 = 2000 if is_armed else 1000
        aux2 = 2000 if is_surface_mode else 1000
        thr_to_send = rc_throttle if is_armed else 1000
        

        channels = [rc_roll, rc_pitch, thr_to_send, rc_yaw, aux1, aux2, 1000, 1000]
        drone.set_rc(channels)
        
        time.sleep(0.02) # 50Hz

def main():
    global rc_throttle, rc_roll, rc_pitch, rc_yaw, is_armed, is_surface_mode, running

    drone = DroneLink()
    t = threading.Thread(target=drone_loop, args=(drone,))
    t.start()

    print("\r\n===============================================")
    print("\r   FULL CONTROL STATION")
    print("\r===============================================")
    print("\r [ t ] -> AUTO START (Procedura Skoku)")
    print("\r [ z ] -> PRZEŁĄCZ TRYB / RATUNEK (1350)")
    print("\r")
    print("\r [ w / s ] -> Gaz +/- (Wznoszenie)")
    print("\r [ h ]     -> Gaz HOLD (Ustaw 1500 - Zwis)")
    print("\r")
    print("\r [ i / k ] -> Przód / Tył")
    print("\r [ j / l ] -> Lewo / Prawo")
    print("\r [ r ]     -> RESET KIERUNKÓW (Stój)")
    print("\r")
    print("\r [ SPACJA] -> KILL SWITCH")
    print("\r===============================================")

    try:
        while running:
            key = getch()
            if key == 't':
                if not is_armed:
                    print("\r\n>>> 1. UZBRAJANIE...")
                    is_surface_mode = False
                    rc_throttle = 1000
                    rc_roll = 1500; rc_pitch = 1500
                    is_armed = False; time.sleep(0.2)
                    is_armed = True
                    
                    time.sleep(1.5)
                    
                    print(f"\r>>> 2. SKOK! (Gaz: {JUMP_THROTTLE})")
                    rc_throttle = JUMP_THROTTLE
                    time.sleep(0.6) 
                    
                    print("\r>>> 3. KOTWICA ON (Gaz: 1500)")
                    rc_throttle = 1500
                    is_surface_mode = True

            elif key == 'z':
                if is_armed:
                    is_surface_mode = not is_surface_mode
                    
                    
                    rc_roll = 1500
                    rc_pitch = 1500
                    
                    if is_surface_mode:
                        rc_throttle = 1500 
                        print("\r\n>>> [AUTO] KOTWICA WŁĄCZONA.")
                    else:
                        rc_throttle = SAFE_HOVER_THROTTLE 
                        print(f"\r\n>>> [MANUAL] PRZEJĘCIE! Gaz ustawiony na {rc_throttle}.")


            elif key == 'w':
                if is_armed and rc_throttle < MAX_THROTTLE:
                    rc_throttle += STEP_THR
                    print(f"\rGaz: {rc_throttle}   ", end='')
            elif key == 's':
                if is_armed and rc_throttle > 1000:
                    rc_throttle -= STEP_THR
                    print(f"\rGaz: {rc_throttle}   ", end='')
            elif key == 'h': 
                rc_throttle = 1500
                print(f"\rGaz: {rc_throttle} (HOLD)", end='')
            elif key == 'i': rc_pitch += STEP_POS; print(f"\rPitch: {rc_pitch}", end='')
            elif key == 'k': rc_pitch -= STEP_POS; print(f"\rPitch: {rc_pitch}", end='')
            elif key == 'j': rc_roll -= STEP_POS;  print(f"\rRoll: {rc_roll}", end='')
            elif key == 'l': rc_roll += STEP_POS;  print(f"\rRoll: {rc_roll}", end='')
            elif key == 'r': 
                rc_roll = 1500
                rc_pitch = 1500
                print("\rRESET kierunków (1500).")

            elif key == ' ' or key == 'x':
                is_armed = False; is_surface_mode = False; rc_throttle = 1000
                print("\r\n!!! DISARM !!!")
            
            elif key == 'q':
                running = False
                is_armed = False

    except Exception as e:
        print(f"\r\nBłąd: {e}")
    finally:
        running = False
        t.join()
        drone.close()
        print("\rKoniec.")

if __name__ == "__main__":
    main()
