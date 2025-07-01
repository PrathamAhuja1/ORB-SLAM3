#!/usr/bin/env python
import rclpy
from rclpy.qos import qos_profile_system_default

from std_msgs.msg import Float64

import sys, select, termios, tty , os
from threading import Lock

if os.name == 'nt':
    import msvcrt
else:
    import termios
    import tty

LRAUV_MAX_THRUST = 10.0
LRAUV_MIN_THRUST = -10.0
LRAUV_MAX_RUDDER = 7.5


msg = """
Control Your AUV!
---------------------------
Surge control:
    j    k

j/k : Surge forward / Surge backward

---------------------------

Moving around:
        w
   a    s    d

w/s : elevate up/elevate down (Pitch Control)
a/d : rudder left/rudder right (Yaw COntrol)

x : force stop

CTRL-C to quit
"""


class Controller:
    def __init__(self):
        self.controllerMutex = Lock()

        self.flthruster = 0
        self.frthruster = 0
        self.blthruster = 0
        self.brthruster = 0
        self.dthruster1 = 0
        self.dthruster2 = 0
    
    def PrintState(self):
        print('Currently:\tflthruster {0}\t frthruster {1}\t blthruster {2}\t brthruster {3}\t dthruster1 {4}\t dthruster2 {5} '.format(
        self.flthruster,
        self.frthruster,
        self.blthruster,
        self.brthruster,
        self.dthruster1,
        self.dthruster2))

def get_key(settings):
    if os.name == 'nt':
        return msvcrt.getch().decode('utf-8')
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''

    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


def main(args=None):    
    
    settings = None
    if os.name != 'nt':
        settings = termios.tcgetattr(sys.stdin)
    control = Controller()

    rclpy.init(args = args)
    node = rclpy.create_node('bluerov_teleop_ign_node')

    flthrusterTopic = "/model/orca4_ign/joint/thruster1_joint/cmd_pos"
    flthrusterPub = node.create_publisher(Float64, flthrusterTopic , qos_profile_system_default)

    frthrusterTopic = "/model/orca4_ign/joint/thruster2_joint/cmd_pos"
    frthrusterPub = node.create_publisher(Float64 , frthrusterTopic , qos_profile_system_default)

    blthrusterTopic = "/model/orca4_ign/joint/thruster3_joint/cmd_pos"
    blthrusterPub = node.create_publisher(Float64 , blthrusterTopic , qos_profile_system_default)

    brthrusterTopic = "/model/orca4_ign/joint/thruster4_joint/cmd_pos"
    brthrusterPub = node.create_publisher(Float64 , brthrusterTopic , qos_profile_system_default)

    dthruster1Topic = "/model/orca4_ign/joint/thruster5_joint/cmd_pos"
    dthruster1Pub = node.create_publisher(Float64 , dthruster1Topic , qos_profile_system_default)

    dthruster2Topic = "/model/orca4_ign/joint/thruster6_joint/cmd_pos"
    dthruster2Pub = node.create_publisher(Float64 , dthruster2Topic , qos_profile_system_default)

    status = 1

    print(msg)

    while True:
        with control.controllerMutex:
            key = get_key(settings)

            if status % 10 == 0 :
                print(msg)

            if(key == 'k'):
                surgeForwardMsg = Float64()
                surgeForwardMsg2 = Float64()
                surgeForwardMsg.data = (LRAUV_MAX_THRUST)
                surgeForwardMsg2.data = (LRAUV_MIN_THRUST)
                flthrusterPub.publish(surgeForwardMsg)
                frthrusterPub.publish(surgeForwardMsg)
                blthrusterPub.publish(surgeForwardMsg2)
                brthrusterPub.publish(surgeForwardMsg2)
                print(surgeForwardMsg)
                control.PrintState()
                status += 1

            if(key == 'j'):
                surgeForwardMsg = Float64()
                surgeForwardMsg2 = Float64()
                surgeForwardMsg.data = (LRAUV_MIN_THRUST)
                surgeForwardMsg2.data = (LRAUV_MAX_THRUST)
                flthrusterPub.publish(surgeForwardMsg)
                frthrusterPub.publish(surgeForwardMsg)
                blthrusterPub.publish(surgeForwardMsg2)
                brthrusterPub.publish(surgeForwardMsg2)
                print(surgeForwardMsg)
                control.PrintState()
                status += 1

            if(key == 'w'):
                heaveUpwardMsg = Float64()
                heaveUpwardMsg.data = (LRAUV_MIN_THRUST)
                dthruster1Pub.publish(heaveUpwardMsg)
                dthruster2Pub.publish(heaveUpwardMsg)
                print(heaveUpwardMsg)
                control.PrintState()
                status += 1

            if(key == 's'):
                heaveDownwardMsg = Float64()
                heaveDownwardMsg.data = (LRAUV_MAX_THRUST)
                dthruster1Pub.publish(heaveDownwardMsg)
                dthruster2Pub.publish(heaveDownwardMsg)
                print(heaveDownwardMsg)
                control.PrintState()
                status += 1

            if(key == 'd'):
                surgeForwardMsg = Float64()
                surgeForwardMsg2 = Float64()
                surgeForwardMsg.data = (LRAUV_MAX_THRUST)
                surgeForwardMsg2.data = (LRAUV_MIN_THRUST)
                flthrusterPub.publish(surgeForwardMsg)
                frthrusterPub.publish(surgeForwardMsg2)
                blthrusterPub.publish(surgeForwardMsg2)
                brthrusterPub.publish(surgeForwardMsg)
                print(surgeForwardMsg)
                control.PrintState()
                status += 1

            if(key == 'a'):
                surgeForwardMsg = Float64()
                surgeForwardMsg2 = Float64()
                surgeForwardMsg.data = (LRAUV_MAX_THRUST)
                surgeForwardMsg2.data = (LRAUV_MIN_THRUST)
                flthrusterPub.publish(surgeForwardMsg2)
                frthrusterPub.publish(surgeForwardMsg)
                blthrusterPub.publish(surgeForwardMsg)
                brthrusterPub.publish(surgeForwardMsg2)
                print(surgeForwardMsg)
                control.PrintState()
                status += 1

            if(key == 'x'):
                stopMsg = Float64()
                stopMsg.data =  0.0
                flthrusterPub.publish(stopMsg)
                frthrusterPub.publish(stopMsg)
                blthrusterPub.publish(stopMsg)
                brthrusterPub.publish(stopMsg)
                dthruster1Pub.publish(stopMsg)
                dthruster2Pub.publish(stopMsg)
                control.PrintState()
                status += 1

            else:
                if (key == '\x03'):
                    break

if __name__ == "__main__":
    main()

