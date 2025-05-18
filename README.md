PyPhysLab: Simple 2D/3D Physics Simulator
A lightweight physics simulator demonstrating particle dynamics in both 2D and 3D environments using Python and Matplotlib. This project serves as an educational tool to visualize basic physical principles like Newton's laws, momentum conservation, and elastic collisions.
Overview
This project contains two main components:

2D Physics Engine: Simulates particles moving in a plane with collision physics
3D Physics Engine: Extends the simulation to three dimensions

Both engines demonstrate:

Vector operations
Particle dynamics with mass, position, and velocity
Force application (including gravity)
Collision detection and response
Real-time visualization

Usage
The project is designed to be used in Jupyter notebooks. Simply copy the code into your notebook cells and run them to see the simulations in action.
Example (2D):
python# Create a physics engine
engine = PhysicsEngine(width=20, height=15)

# Add a few particles
engine.add_particle(Particle(5.0, Vector2D(10, 7.5), Vector2D(0, 0), 1.0, 'red'))
engine.add_particle(Particle(1.0, Vector2D(5, 10), Vector2D(2, 0), 0.5, 'blue'))

# Run the simulation
engine.visualize(simulation_time=10)
Example (3D):
python# Create a 3D physics engine
engine = PhysicsEngine3D(width=20, height=20, depth=20)

# Add particles
engine.add_particle(Particle3D(10.0, Vector3D(10, 10, 10), Vector3D(0, 0, 0), 1.5, 'red'))
engine.add_particle(Particle3D(1.0, Vector3D(5, 5, 15), Vector3D(2, 1, 0), 0.7, 'blue'))

# Run the simulation
engine.visualize(simulation_time=15)
Requirements

Python 3.x
NumPy
Matplotlib
IPython (for Jupyter notebook integration)


This project was created for educational purposes to help visualize physics concepts through interactive simulations.
