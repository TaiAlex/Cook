import pygame
import math
import numpy as np
from scipy.special import expit
import time
import pandas as pd


WHITE_COLOR = (255, 255, 255)
GREEN_COLOR = (0, 255, 0)
RED_COLOR = (255, 0, 0)

class Envir:
    def __init__(self, dim):
        self.black = (0,0,0)
        self.white = (255,255,255)
        self.red = (255,0,0)
        self.green = (0,255,0)
        self.blue = (0,0,255)
        self.height = dim[0]
        self.width = dim[1]
        
        self.edge_distances = [0,0,0,0,0,0]
        
        pygame.display.set_caption('Cooking ROBOT')
        self.map = pygame.display.set_mode((self.width, self.height))
        
        self.font = pygame.font.SysFont('arial', 30)
        self.text = self.font.render('default', True, self.black, self.white)
        self.textRect = self.text.get_rect()
        self.textRect.center = (dim[1] - 700, dim[0] - 100)
        self.trail_set = []
        
    def sensor_info(self, sensor_data):
        text  = f"sensor: {sensor_data}"
        self.text2 = self.font.render(text, True, self.black, self.white)
        self.textRect.center = (self.width + 700, self.height - 50)
        self.map.blit(self.text2, self.textRect)
    
    def info(self, gen, t1, t2, t3):
        text = f'Generation: {gen}, Time: {t1}, pJbest: {t2},fitness: {t3}'
        self.text = self.font.render(text, True, self.black, self.white)
        self.textRect.center = (self.width - 700, self.height - 50)
        self.map.blit(self.text, self.textRect)
        
    def trail(self, pos):
        for i in range(0, len(self.trail_set) - 1):
            pygame.draw.line(self.map,self.red,(self.trail_set[i][0],self.trail_set[i][1]),(self.trail_set[i+1][0],self.trail_set[i+1][1]))
        if self.trail_set.__sizeof__() > 30000:
            self.trail_set.pop(0)
        self.trail_set.append(pos)
        
    def robot_frame(self, pos, rotation):
        n = 80
        centerx, centery = pos
        x_axis = (centerx + n*math.cos(rotation), centery + n*math.sin(rotation))
        y_axis = (centerx + n*math.cos(rotation + math.pi/2), centery + n*math.sin(rotation + math.pi/2))
        pygame.draw.line(self.map, self.blue, pos, x_axis, 3)
        pygame.draw.line(self.map, self.red, pos, y_axis, 3)
    
    def robot_sensor(self, pos, points):
        for point in points:
            pygame.draw.line(self.map, (0,255,0), pos, point)
            pygame.draw.circle(self.map, (0, 255, 0), point, 5)
        
class Robot:
    def __init__(self, startpos, Img, width):
        self.w = width
        self.x = startpos[0]
        self.y = startpos[1]
        self.theta = 0
        self.v1 = 0
        self.v2 = 0
        self.v3 = 0
        self.vx = 0
        self.vy = 0
        self.theta_dot = 0
        self.vxg = 0
        self.vyg = 0
        self.theta_d = 0
        self.sensor_data = [0,0,0,0,0,0]
        self.points = []
        self.crash = False
        self.finish = False
        self.cost_function = 0
        
        self.img = pygame.image.load(Img)
        self.img = pygame.transform.scale(self.img, (50,50))
        self.rotated = self.img
        self.rect = self.rotated.get_rect(center = (self.x, self.y))
        
    def update_sensor_data(self):
        angles = [self.theta, np.pi/3 + self.theta, 2*np.pi/3 + self.theta,
                  np.pi + self.theta, 4*np.pi/3 + self.theta, 5*np.pi/3 + self.theta]
        
        edge_points = []
        edge_distances = []
        for angle in angles:
            distance = 0
            edge_x, edge_y = (int(self.x), int(self.y))
            while track_copy.get_at((edge_x, edge_y)) != WHITE_COLOR:
                edge_x = int(self.x + distance * math.cos(angle))
                edge_y = int(self.y + distance * math.sin(angle))
                distance += 1
            edge_points.append((edge_x, edge_y))
            edge_distances.append(distance)
        self.sensor_data = edge_distances
        self.points = edge_points
        
    def check_crash(self):
        edge_x, edge_y = (int(self.x), int(self.y))
        if track_copy.get_at((edge_x, edge_y)) == WHITE_COLOR or track_copy.get_at((edge_x, edge_y)) == RED_COLOR:
            self.crash = True
            self.cost_function = ((self.x) - end_point[0])**2 + ((self.y) - end_point[1])**2
        if track_copy.get_at((edge_x, edge_y)) == GREEN_COLOR:
            self.finish = True
        
    def draw(self, map):
        map.blit(self.rotated, self.rect)
        
    def neuron(self, X, V, W):
        net_h = V.T @ X
        y_h = expit(net_h)
        net_0 = W.T @ y_h
        y = net_0
        return y
        
    def move(self, event = None):
        inv_R = np.array([  [np.cos(self.theta), -np.sin(self.theta), 0],
                            [np.sin(self.theta), np.cos(self.theta), 0],
                            [0, 0, 1]])
        l = 10
        inv_j1 = np.array([[1/np.sqrt(3), 0, -1/np.sqrt(3)], [-1/3, 2/3, -1/3], [-1/(3*l), -1/(3*l), -1/(3*l)]])
        r = 3
        j2 = np.array([[r, 0, 0], [0, r, 0], [0, 0, r]])
        V = np.linalg.inv(j2) @ np.linalg.inv(inv_j1) @ np.array([[self.vxg], [self.vyg], [self.theta_d]])
        self.v1 = V[0, 0]
        self.v2 = V[1, 0]
        self.v3 = V[2, 0]
        AAA = inv_R @ inv_j1 @ j2 @ np.array([[self.v1], [self.v2], [self.v3]])
        self.vx = AAA[0, 0]
        self.vy = AAA[1, 0]
        self.theta_dot = AAA[2, 0]
        
        self.x = self.x + self.vx*dt
        self.y = self.y + self.vy*dt
        self.theta = self.theta + self.theta_dot*dt
        self.rotated = pygame.transform.rotozoom(self.img, -math.degrees(self.theta), 1)
        self.rect = self.rotated.get_rect(center = (self.x, self.y))
    
pygame.init()
pygame.display.set_mode((1280, 720))
track = pygame.image.load("track2.png")
track_copy = track.copy()

start = (200, 300)
end_point = (1090, 610)
dims = (720, 1280)
running = True
dt = 0
lasttime = pygame.time.get_ticks()
environment = Envir(dims)
# Số thế hệ bạn muốn tạo
N = 20
npar = 15
min_c = -1
max_c = 1
max_iteration = 20
c1 = 0.1
c2 = 0.1
P = min_c * np.ones((N, npar)) + (max_c - min_c) * np.random.rand(N, npar)
V1 = np.zeros((N, npar))
J = np.zeros((N, 1))
gJbest = 999999
robot1 = Robot(start, "delta1.png", 1)
robot1.theta_d = np.pi/2
while running:
    for generation in range(max_iteration):
        pJbest = 999999
        # Tạo thế hệ đầu tiên với 5 robot
        robots = []
        second = time.strftime("%S")
        minute = time.strftime("%M")
        tim = 60*int(minute) + int(second)
        for rb in range(N):
            robot = Robot(start, "delta1.png", 1)
            robot.theta_d = np.pi/2
            robots.append(robot)
        temps = robots.copy()
        print(f'Generation: {generation+1}')
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            second1 = time.strftime("%S")
            minute1 = time.strftime("%M")
            time1 = 60*int(minute1) + int(second1)
            delta = time1 - tim
            if len(temps) == 0:
                df = pd.DataFrame(P)
                df.to_csv("best.csv")
                temps.clear()
                V1 = V1 + c1 * np.random.random() * (pbest - P) + c2 * np.random.random() * (gbest - P)
                P = P + V1
                break
            if delta >= 60:
                temps.clear()
                V1 = V1 + c1 * np.random.random() * (pbest - P) + c2 * np.random.random() * (gbest - P)
                P = P + V1
            for temp in temps:
                if temp.finish == True or temp.crash == True:
                    temps.remove(temp)
                if temp in robots:
                    j = robots.index(temp)
                V = P[j, 0:10].reshape(2, 5)
                W = P[j, 10:15].reshape(5, 1)
                temp.vxg = 40
                temp.theta_d = temp.neuron(np.array([[temp.sensor_data[5]], [temp.sensor_data[1]]]), V, W)[0, 0]
                temp.move()
                temp.cost_function  = ((temp.x) - end_point[0])**2 + ((temp.y) - end_point[1])**2 + (temp.theta - np.pi/2)**2
                environment.robot_frame((temp.x, temp.y), temp.theta)
                environment.robot_sensor((temp.x, temp.y), temp.points)
                environment.info(generation + 1, delta%100, int(pJbest), int(gJbest))
                temp.check_crash()
                temp.update_sensor_data()
                temp.draw(environment.map)
                J[j] = temp.cost_function
                if (J[j] < pJbest):
                    pJbest = J[j]
                    pbest = P[j,:]
            if (pJbest < gJbest):
                # print("Change")
                gJbest = pJbest
                gbest = pbest
            dt = (pygame.time.get_ticks() - lasttime)/1000
            lasttime = pygame.time.get_ticks()
            pygame.display.update()
            environment.map.blit(track, (0, 0))
    # break
# V = gbest[0:10].reshape(2, 5)
# W = gbest[10:15].reshape(5, 1)
# for gen in range(max_iteration):
#     robot1.vxg = 100
#     robot1.theta_d = robot1.neuron(np.array([[robot1.sensor_data[5]], [robot1.sensor_data[1]]]), V, W)[0, 0]
#     robot1.move()
#     robot1.cost_function  = ((robot1.x) - end_point[0])**2 + ((robot1.y) - end_point[1])**2 + (robot1.theta - np.pi/2)**2
#     environment.robot_frame((robot1.x, robot1.y), robot1.theta)
#     environment.robot_sensor((robot1.x, robot1.y), robot1.points)
#     environment.info(generation + 1, delta%100, int(pJbest), int(gJbest))
#     robot1.check_crash()
#     robot1.update_sensor_data()
#     robot1.draw(environment.map)
#     e = robot1.cost_function
#     dt = (pygame.time.get_ticks() - lasttime)/1000
#     lasttime = pygame.time.get_ticks()
#     pygame.display.update()
#     environment.map.blit(track, (0, 0))
