import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Point
import numpy as np
import math
from stable_baselines3 import PPO  # O PPO, TD3, ecc.
from stable_baselines3.common.vec_env import VecNormalize
import torch


class RLTurtleBotNode(Node):
    def __init__(self):
        super().__init__('rl_turtlebot_node')

        # Publisher velocità
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Nuovo subscriber per leggere da un topic il punto da raggiungere
        self.create_subscription(Point, '/target_position', self.target_pos_callback, 10)

        # Subscriber laser e odometria
        self.create_subscription(LaserScan, '/scan', self.laser_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        # Timer per inference
        self.timer = self.create_timer(0.1, self.control_loop)  # 10Hz

        # Stato
        self.laser = None
        self.position = None
        self.yaw = None
        self.target_pos = np.array([0.0, 0.0])

        # Caricamento modello
        self.model = PPO.load("ppo_model.zip")  # Percorso del modello
        try:
            self.vecnorm = VecNormalize.load("vecnormalize.pkl", None)
            self.vecnorm.training = False
            self.vecnorm.norm_reward = False
        except:
            self.vecnorm = None

        self.get_logger().info("✅ RL controller ready")

    def target_pos_callback(self, msg: Point):
        self.target_pos = np.array([msg.x, msg.y])
        self.get_logger().info(f"Target position updated: {self.target_pos}")

    def odom_callback(self, msg):
        # Posizione XY
        self.position = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y
        ])

        # Estrazione yaw da quaternion
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.yaw = math.atan2(siny_cosp, cosy_cosp)

    def laser_callback(self, msg: LaserScan):
        # Salva laser scan decimato
        ranges = np.array(msg.ranges)
        
        laser = ranges[::5] # Decimazione ogni 5 misure (simulazione → 1080/5=216)


        # Prendi solo l'intervallo 0–180° (cioè [0, π])
        angles = np.linspace(msg.angle_min, msg.angle_max, len(ranges)) # Genera gli angoli
        front_indices = np.where((angles >= 0) & (angles <= math.pi))[0]

        # Filtra i dati del laser per l'intervallo frontale
        self.laser = laser[front_indices]

        

    def compute_observation(self):
        if self.position is None or self.yaw is None or self.laser is None:
            return None

        # Calcola distanza e azimuth relativi al target
        delta = self.target_pos - self.position
        distance = np.linalg.norm(delta)
        target_angle = math.atan2(delta[1], delta[0])
        relative_azimuth = target_angle - self.yaw
        relative_azimuth = (relative_azimuth + np.pi) % (2 * np.pi) - np.pi

        # Osservazione simulata: [laser..., distance, azimuth]
        obs = np.concatenate([self.laser, [distance, relative_azimuth]]).astype(np.float32)
        obs = obs.reshape(1, -1)

        if self.vecnorm:
            obs = self.vecnorm.normalize_obs(obs)

        return obs

    def control_loop(self):
        obs = self.compute_observation()
        if obs is None:
            return

        action, _ = self.model.predict(obs, deterministic=True)

        # Decodifica azione in velocità
        lin_vel = float(action[0])  # max linear velocity TurtleBot4
        ang_vel = float(action[1])  # max angular velocity

        # Pubblica su /cmd_vel
        cmd = Twist()
        cmd.linear.x = lin_vel
        cmd.angular.z = ang_vel
        self.cmd_vel_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = RLTurtleBotNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
