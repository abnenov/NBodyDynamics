import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

class Vector2D:
    """
    Клас - 2D вектор с операции
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __add__(self, other):
        """Събиране на вектори"""
        return Vector2D(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        """Изваждане на вектори"""
        return Vector2D(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar):
        """Умножение на вектор по скалар"""
        return Vector2D(self.x * scalar, self.y * scalar)
    
    def __rmul__(self, scalar):
        """Умножение на скалар по вектор (от дясно)"""
        return self.__mul__(scalar)
    
    def __truediv__(self, scalar):
        """Деление на вектор на скалар"""
        return Vector2D(self.x / scalar, self.y / scalar)
    
    def dot(self, other):
        """Скаларно произведение"""
        return self.x * other.x + self.y * other.y
    
    def magnitude(self):
        """Големина (модул) на вектора"""
        return np.sqrt(self.x**2 + self.y**2)
    
    def normalize(self):
        """Нормализиране на вектора (единичен вектор в същата посока)"""
        mag = self.magnitude()
        if mag == 0:
            return Vector2D(0, 0)
        return Vector2D(self.x / mag, self.y / mag)
    
    def __str__(self):
        return f"Vector2D({self.x}, {self.y})"
    
    def to_tuple(self):
        """Превръща вектора в tuple за ползване с matplotlib"""
        return (self.x, self.y)


class Particle:
    """
    Клас, представляващ материална точка (частица) с маса, позиция и скорост
    """
    def __init__(self, mass, position, velocity, radius=0.5, color='blue'):
        self.mass = mass
        self.position = position  # Vector2D
        self.velocity = velocity  # Vector2D
        self.acceleration = Vector2D(0, 0)
        self.forces = []
        self.radius = radius
        self.color = color
    
    def apply_force(self, force):
        """Прилага сила към частицата"""
        self.forces.append(force)
    
    def clear_forces(self):
        """Изчиства всички сили"""
        self.forces = []
    
    def update(self, dt):
        """Обновява състоянието на частицата за времеви интервал dt"""
        # Изчисляване на резултантната сила
        resultant_force = Vector2D(0, 0)
        for force in self.forces:
            resultant_force = resultant_force + force
        
        # F = ma => a = F/m (Втори закон на Нютон)
        self.acceleration = resultant_force / self.mass
        
        # Обновяване на скоростта: v = v0 + a*dt
        self.velocity = self.velocity + self.acceleration * dt
        
        # Обновяване на позицията: s = s0 + v*dt
        self.position = self.position + self.velocity * dt
        
        # Изчистване на силите за следващата итерация
        self.clear_forces()


class PhysicsEngine:
    """
    Физически двигател, управляващ симулацията
    """
    def __init__(self, width=10, height=10):
        self.particles = []
        self.width = width
        self.height = height
        self.gravity = Vector2D(0, -9.81)  # Гравитационно ускорение
        self.elasticity = 0.8  # Коефициент на еластичност при сблъсъци
        
    def add_particle(self, particle):
        """Добавя частица към симулацията"""
        self.particles.append(particle)
    
    def apply_gravity(self):
        """Прилага гравитационна сила към всички частици"""
        for particle in self.particles:
            # F = mg
            gravitational_force = self.gravity * particle.mass
            particle.apply_force(gravitational_force)
    
    def check_boundary_collisions(self):
        """Проверява за сблъсъци със стените на симулацията"""
        for particle in self.particles:
            # Проверка за долна стена
            if particle.position.y - particle.radius < 0:
                # Коригиране на позицията
                particle.position.y = particle.radius
                # Обръщане на вертикалната компонента на скоростта
                particle.velocity.y = -particle.velocity.y * self.elasticity
            
            # Проверка за горна стена
            if particle.position.y + particle.radius > self.height:
                particle.position.y = self.height - particle.radius
                particle.velocity.y = -particle.velocity.y * self.elasticity
            
            # Проверка за лява стена
            if particle.position.x - particle.radius < 0:
                particle.position.x = particle.radius
                particle.velocity.x = -particle.velocity.x * self.elasticity
            
            # Проверка за дясна стена
            if particle.position.x + particle.radius > self.width:
                particle.position.x = self.width - particle.radius
                particle.velocity.x = -particle.velocity.x * self.elasticity
    
    def check_particle_collisions(self):
        """Проверява за сблъсъци между частиците"""
        num_particles = len(self.particles)
        
        for i in range(num_particles):
            for j in range(i+1, num_particles):
                p1 = self.particles[i]
                p2 = self.particles[j]
                
                # Изчисляване на вектора между центровете на частиците
                distance_vector = p2.position - p1.position
                distance = distance_vector.magnitude()
                
                # Проверка за сблъсък
                min_distance = p1.radius + p2.radius
                if distance < min_distance:
                    # Нормализиран вектор на посоката на сблъсъка
                    collision_normal = distance_vector.normalize()
                    
                    # Коригиране на позициите, за да няма препокриване
                    overlap = min_distance - distance
                    p1.position = p1.position - collision_normal * (overlap * 0.5)
                    p2.position = p2.position + collision_normal * (overlap * 0.5)
                    
                    # Изчисляване на относителната скорост по посока на сблъсъка
                    relative_velocity = p2.velocity - p1.velocity
                    relative_speed = relative_velocity.dot(collision_normal)
                    
                    # Ако частиците се отдалечават, не правим нищо
                    if relative_speed > 0:
                        continue
                    
                    # Изчисляване на импулса (закон за запазване на импулса и енергията)
                    impulse_magnitude = -(1 + self.elasticity) * relative_speed
                    impulse_magnitude /= (1/p1.mass + 1/p2.mass)
                    
                    # Прилагане на импулса към скоростите
                    p1.velocity = p1.velocity - impulse_magnitude * collision_normal / p1.mass
                    p2.velocity = p2.velocity + impulse_magnitude * collision_normal / p2.mass
    
    def update(self, dt):
        """Обновява състоянието на всички частици за времеви интервал dt"""
        # Прилагане на гравитация
        self.apply_gravity()
        
        # Обновяване на всички частици
        for particle in self.particles:
            particle.update(dt)
        
        # Проверка за сблъсъци със стените
        self.check_boundary_collisions()
        
        # Проверка за сблъсъци между частиците
        self.check_particle_collisions()
    
    def visualize(self, simulation_time=10, dt=0.01, fps=30):
        """Визуализира симулацията с анимация"""
        # Създаване на фигура и оси
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal')
        ax.set_title('2D Physics Engine Simulation')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        # Списък с кръгове за всяка частица
        circles = []
        for particle in self.particles:
            circle = plt.Circle(particle.position.to_tuple(), particle.radius, color=particle.color)
            circles.append(circle)
            ax.add_patch(circle)
        
        # Брой кадри
        frames = int(simulation_time / dt * (dt * fps))
        
        # Функция за обновяване на анимацията
        def update_animation(frame):
            # Обновяване на физиката няколко пъти между кадрите за по-плавна симулация
            for _ in range(int(1/(dt * fps))):
                self.update(dt)
            
            # Обновяване на позициите на кръговете
            for i, particle in enumerate(self.particles):
                circles[i].center = particle.position.to_tuple()
            
            return circles
        
        # Създаване и показване на анимацията
        animation = FuncAnimation(fig, update_animation, frames=frames, interval=1000/fps, blit=True)
        plt.close()  # Да не показва фигурата два пъти
        return HTML(animation.to_jshtml())


# Примерна употреба
def run_simulation():
    # Създаване на физически двигател
    engine = PhysicsEngine(width=20, height=15)
    
    # Добавяне на няколко частици
    # Параметри: маса, начална позиция, начална скорост, радиус, цвят
    
    # Тежка частица в центъра
    engine.add_particle(Particle(5.0, Vector2D(10, 7.5), Vector2D(0, 0), 1.0, 'red'))
    
    # Няколко по-малки частици в различни позиции
    engine.add_particle(Particle(1.0, Vector2D(5, 10), Vector2D(2, 0), 0.5, 'blue'))
    engine.add_particle(Particle(1.0, Vector2D(15, 10), Vector2D(-2, 0), 0.5, 'green'))
    engine.add_particle(Particle(1.0, Vector2D(8, 5), Vector2D(1, 3), 0.5, 'purple'))
    engine.add_particle(Particle(1.0, Vector2D(12, 5), Vector2D(-1, 3), 0.5, 'orange'))
    
    # Много малка частица с голяма скорост
    engine.add_particle(Particle(0.5, Vector2D(3, 3), Vector2D(5, 5), 0.3, 'cyan'))
    
    # Запускане на симулацията за 10 секунди
    return engine.visualize(simulation_time=10, dt=0.01, fps=30)

# Изпълнение на симулацията
run_simulation()
