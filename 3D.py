import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import HTML

class Vector3D:
    """
    Клас за 3D вектор с операции
    """
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
    
    def __add__(self, other):
        """Събиране на вектори"""
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other):
        """Изваждане на вектори"""
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar):
        """Умножение на вектор по скалар"""
        return Vector3D(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def __rmul__(self, scalar):
        """Умножение на скалар по вектор (от дясно)"""
        return self.__mul__(scalar)
    
    def __truediv__(self, scalar):
        """Деление на вектор на скалар"""
        return Vector3D(self.x / scalar, self.y / scalar, self.z / scalar)
    
    def dot(self, other):
        """Скаларно произведение"""
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def cross(self, other):
        """Векторно произведение"""
        return Vector3D(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
    
    def magnitude(self):
        """Mодул на вектора"""
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def normalize(self):
        """Нормализиране на вектор"""
        mag = self.magnitude()
        if mag == 0:
            return Vector3D(0, 0, 0)
        return Vector3D(self.x / mag, self.y / mag, self.z / mag)
    
    def __str__(self):
        return f"Vector3D({self.x}, {self.y}, {self.z})"
    
    def to_tuple(self):
        """Превръща вектора в tuple"""
        return (self.x, self.y, self.z)


class Particle3D:
    """
    Клас - материална точка/частица в 3D пространството
    """
    def __init__(self, mass, position, velocity, radius=0.5, color='blue'):
        self.mass = mass
        self.position = position  # Vector3D
        self.velocity = velocity  # Vector3D
        self.acceleration = Vector3D(0, 0, 0)
        self.forces = []
        self.radius = radius
        self.color = color
    
    def apply_force(self, force):
        """Прилагане на сила към частицата"""
        self.forces.append(force)
    
    def clear_forces(self):
        """Изчиства действащите сили"""
        self.forces = []
    
    def update(self, dt):
        """Обновява състоянието на частицата за времеви интервал dt"""
        # Изчисляване на резултантната сила
        resultant_force = Vector3D(0, 0, 0)
        for force in self.forces:
            resultant_force = resultant_force + force
        
        # Втори закон на Нютон F = ma => a = F/m 
        self.acceleration = resultant_force / self.mass
        
        # Опресняване на стойността на скоростта: v = v0 + a*dt
        self.velocity = self.velocity + self.acceleration * dt
        
        # Обновяване на позицията: s = s0 + v*dt
        self.position = self.position + self.velocity * dt
        
        # Изчистване на силите за следващата итерация
        self.clear_forces()


class PhysicsEngine3D:
    """
    Триизмерен физически двигател на симулацията
    """
    def __init__(self, width=10, height=10, depth=10):
        self.particles = []
        self.width = width
        self.height = height
        self.depth = depth
        self.gravity = Vector3D(0, 0, -9.81)  # Гравитационно ускорение по оста Z
        self.elasticity = 0.8  # Коефициент на еластичност при сблъсъци
        
    def add_particle(self, particle):
        """Добавяме частица към симулацията"""
        self.particles.append(particle)
    
    def apply_gravity(self):
        """Прилагаме гравитационна сила към всички частици"""
        for particle in self.particles:
            # F = mg
            gravitational_force = self.gravity * particle.mass
            particle.apply_force(gravitational_force)
    
    def check_boundary_collisions(self):
        """Правим проерка за сблъсъци със стените на симулацията"""
        for particle in self.particles:
            # Проверка за долна стена (Z)
            if particle.position.z - particle.radius < 0:
                # Коригиране на позицията
                particle.position.z = particle.radius
                # Обръщане на Z компонента на скоростта
                particle.velocity.z = -particle.velocity.z * self.elasticity
            
            # Проверка за горна стена (Z)
            if particle.position.z + particle.radius > self.depth:
                particle.position.z = self.depth - particle.radius
                particle.velocity.z = -particle.velocity.z * self.elasticity
            
            # Проверка за лява стена (X)
            if particle.position.x - particle.radius < 0:
                particle.position.x = particle.radius
                particle.velocity.x = -particle.velocity.x * self.elasticity
            
            # Проверка за дясна стена (X)
            if particle.position.x + particle.radius > self.width:
                particle.position.x = self.width - particle.radius
                particle.velocity.x = -particle.velocity.x * self.elasticity
                
            # Проверка за предна стена (Y)
            if particle.position.y - particle.radius < 0:
                particle.position.y = particle.radius
                particle.velocity.y = -particle.velocity.y * self.elasticity
            
            # Проверка за задна стена (Y)
            if particle.position.y + particle.radius > self.height:
                particle.position.y = self.height - particle.radius
                particle.velocity.y = -particle.velocity.y * self.elasticity
    
    def check_particle_collisions(self):
        """Проверка за сблъсъци между частиците"""
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
        """Опресняване на състоянието на всички частици за времеви интервал dt"""
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
        """Анимация на 3D симулацията"""
        # Създаване на фигура и 3D оси
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Настройка на осите
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_zlim(0, self.depth)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Physics Engine Simulation')
        
        # Прозрачни стени на куба (опция)
        # Това прави фигурата по-визуално разбираема, но може да натовари визуализацията
        # Да се закоментират тези редове, ако визуализацията е бавна
        
        # Долна стена (z=0)
        xx, yy = np.meshgrid(np.linspace(0, self.width, 2), np.linspace(0, self.height, 2))
        ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.1, color='gray')
        
        # Задна стена (y=height)
        xx, zz = np.meshgrid(np.linspace(0, self.width, 2), np.linspace(0, self.depth, 2))
        ax.plot_surface(xx, np.ones_like(xx) * self.height, zz, alpha=0.1, color='gray')
        
        # Дясна стена (x=width)
        yy, zz = np.meshgrid(np.linspace(0, self.height, 2), np.linspace(0, self.depth, 2))
        ax.plot_surface(np.ones_like(yy) * self.width, yy, zz, alpha=0.1, color='gray')
        
        # Създаване на списък със сфери за всяка частица
        # В matplotlib няма вградена поддръжка за анимация на сфери,
        # Scatter plot с различни размери за частиците
        particles_scatter = ax.scatter(
            [p.position.x for p in self.particles],
            [p.position.y for p in self.particles],
            [p.position.z for p in self.particles],
            s=[p.radius * 100 for p in self.particles],  # Размерите в scatter са в точки²
            c=[p.color for p in self.particles],
            alpha=0.8
        )
        
        # Траектории на частиците (опция)
        # Трябва да се съхранява историята на позициите на всяка частица
        trajectories = [[] for _ in self.particles]
        trajectory_lines = [ax.plot([], [], [], '--', lw=0.5, alpha=0.5, color=p.color)[0] for p in self.particles]
        
        # Текст за време
        time_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)
        
        # Инициализация на анимацията
        def init():
            particles_scatter._offsets3d = ([], [], [])
            for line in trajectory_lines:
                line.set_data([], [])
                line.set_3d_properties([])
            time_text.set_text("")
            return [particles_scatter] + trajectory_lines + [time_text]
        
        # Брой кадри
        frames = int(simulation_time / dt * (dt * fps))
        
        # Обновяване на анимацията
        def update_animation(frame):
            current_time = frame * dt * (1/(dt * fps))
            
            # Обновяване на физиката няколко пъти между кадрите за по-плавна симулация
            for _ in range(int(1/(dt * fps))):
                self.update(dt)
            
            # Обновяване на позициите на частиците в scatter plot
            particles_scatter._offsets3d = (
                [p.position.x for p in self.particles],
                [p.position.y for p in self.particles],
                [p.position.z for p in self.particles]
            )
            
            # Добавяне на позициите към траекториите
            for i, p in enumerate(self.particles):
                trajectories[i].append(p.position.to_tuple())
                
                # Показвам само последните 100 точки от траекторията че да избегна претоварването на моята "барака"
                # Подлежи на регулиране
                if len(trajectories[i]) > 100:
                    trajectories[i] = trajectories[i][-100:]
                
                # Обновяване на линиите на траекториите
                if trajectories[i]:
                    x_traj, y_traj, z_traj = zip(*trajectories[i])
                    trajectory_lines[i].set_data(x_traj, y_traj)
                    trajectory_lines[i].set_3d_properties(z_traj)
            
            # Обновяване на текста за време
            time_text.set_text(f"Time: {current_time:.2f}s")
            
            # Връткаме гледната точка за по-добра 3D визуализация
            # Коментираме този ред за спиране на автоматично въртене
            ax.view_init(elev=30, azim=(frame/10) % 360)
            
            return [particles_scatter] + trajectory_lines + [time_text]
        
        # Самата анимацията
        animation = FuncAnimation(fig, update_animation, frames=frames, interval=1000/fps, blit=False, init_func=init)
        plt.close()  # Да не показва фигурата два пъти
        return HTML(animation.to_jshtml())


def run_3d_simulation():
    # Create 3D физически двигател
    engine = PhysicsEngine3D(width=20, height=20, depth=20)
    
    # ADd на няколко частици
    # Параметри: маса, начална позиция (x,y,z), начална скорост (vx,vy,vz), радиус, цвят
    
    # Тежка частица в центъра
    engine.add_particle(Particle3D(10.0, Vector3D(10, 10, 10), Vector3D(0, 0, 0), 1.5, 'red'))
    
    # Оше Няколко по-малки частици на различни позиции
    engine.add_particle(Particle3D(1.0, Vector3D(5, 5, 15), Vector3D(2, 1, 0), 0.7, 'blue'))
    engine.add_particle(Particle3D(1.0, Vector3D(15, 5, 15), Vector3D(-2, 1, 0), 0.7, 'green'))
    engine.add_particle(Particle3D(1.0, Vector3D(5, 15, 15), Vector3D(2, -1, 0), 0.7, 'purple'))
    engine.add_particle(Particle3D(1.0, Vector3D(15, 15, 15), Vector3D(-2, -1, 0), 0.7, 'orange'))
    
    # Частици с различни вертикални скорости
    engine.add_particle(Particle3D(0.7, Vector3D(8, 8, 18), Vector3D(1, 1, -2), 0.5, 'cyan'))
    engine.add_particle(Particle3D(0.7, Vector3D(12, 12, 5), Vector3D(-1, -1, 3), 0.5, 'magenta'))
    
    # "Дъжд" от частици в горната част на куба
    for i in range(15):
        x = 2 + (i % 5) * 4
        y = 2 + (i // 5) * 8
        engine.add_particle(Particle3D(0.3, Vector3D(x, y, 19), Vector3D(0, 0, -1), 0.3, 'yellow'))
    
    # Запускане на симулацията за 15 секунди
    return engine.visualize(simulation_time=15, dt=0.01, fps=30)

# Изпълнение на симулацията
run_3d_simulation()
