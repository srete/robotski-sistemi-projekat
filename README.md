# Contact Dynamics Simulator

This project was developed as part of the course [Robotics Systems](https://www.etf.bg.ac.rs/fis/karton_predmeta/13M051RS-2019), academic year 2024/2025.  

## Description

The simulator implements rigid-body contact dynamics with friction using the **Staggered Projections (SP)** algorithm [Kaufman et al., 2008].  

It combines:
- Contactless dynamics using rigid-body equations of motion,
- Reference tracking with a PD controller,
- Frictional contact forces modeled with staggered projections.

The main class implementing the simulator is in **`simulator.py`**.  
Examples can be run from **`examples.ipynb`**, which demonstrates the following scenes:
- Three Cubes Falling  
- House of Cards  
- Robotic Hand Moving Fingers  
- Robotic Hand Manipulating Cubes  
- Ball and Boxes Collision  
- Quadruped on Tilted Floor  

The implementation uses the **Pinocchio** library [Carpentier et al., 2019] for dynamics and Jacobian computations, and **ProxQP** [Bambade et al., 2025] for quadratic programming in contact optimization.

---

## References

- Carpentier, J., Saurel, G., Buondonno, G., Mirabel, J., Lamiraux, F., Stasse, O., & Mansard, N. (2019).  
  *The Pinocchio C++ library -- A fast and flexible implementation of rigid body dynamics algorithms and their analytical derivatives*.  
  In *IEEE International Symposium on System Integrations (SII)*.  

- Kaufman, D. M., Sueda, S., James, D. L., & Pai, D. K. (2008).  
  *Staggered Projections for Frictional Contact in Multibody Systems*.  
  In *ACM SIGGRAPH 2008 papers*.  

- Bambade, A., Schramm, F., El-Kazdadi, S., Caron, S., Taylor, A., & Carpentier, J. (2025).  
  *ProxQP: an Efficient and Versatile Quadratic Programming Solver for Real-Time Robotics Applications and Beyond*.  
  *IEEE Transactions on Robotics*.  
