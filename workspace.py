import numpy as np
import matplotlib.pyplot as plt

la, lb, lc = 0.3, 0.339, 0.087

def cinematique(theta1, theta4): 
    """
    Calcule la position cartésienne (xc, yc) (m) de la manette à partir des angles moteurs theta1 et theta4 (rad).
    
    Basé sur la géométrie du mécanisme à 5 barres symétrique.
    - theta1 : angle du bras gauche (rad)
    - theta4 : angle du bras droit (rad)
    - lb, la, lc : longueurs des barres du mécanisme (variables globales) (m)
    
    Retourne :
        xc, yc : coordonnées XY de la manette (m)
    """
    E = 2 * lb * (lc + la * (np.cos(theta4) - np.cos(theta1)))
    F = 2 * la * lb * (np.sin(theta4) - np.sin(theta1))
    G = lc**2 + 2 * la**2 + 2 * lc * la * np.cos(theta4) - 2 * lc * la * np.cos(theta1) - 2 * la**2 * np.cos(theta4 - theta1)
    disc = E**2 + F**2 - G**2
    if disc < 0:
        return None
    angle = 2 * np.arctan((-F - np.sqrt(disc)) / (G - E))
    xc = lc + la * np.cos(theta4) + lb * np.cos(angle)
    yc = la * np.sin(theta4) + lb * np.sin(angle)
    return xc, yc

def cinematique_inverse(xc, yc): 
    """
    Calcule les angles moteurs (theta1, theta4) (rad) nécessaires pour atteindre un point (xc, yc) (m) dans le plan.
    
    Inverse de la fonction cinématique directe.
    
    Retourne :
        theta1, theta4 : angles (rad)
    """
    # Équations géométriques pour les deux côtés du mécanisme
    E1 = -2 * la * xc
    F1 = -2 * la * yc
    G1 = la**2 - lb**2 + xc**2 + yc**2 

    E4 = 2 * la * (-xc + lc)
    F4 = -2 * la * yc
    G4 = lc**2 + la**2 - lb**2 + xc**2 + yc**2 - 2 * lc * xc

    # Résolution par la méthode de l’arc tangente double
    theta1 = 2 * np.atan((-F1 + np.sqrt(E1**2 + F1**2 - G1**2)) / (G1 - E1))
    theta4 = 2 * np.atan((-F4 - np.sqrt(E4**2 + F4**2 - G4**2)) / (G4 - E4))
    
    return theta1, theta4

def jacobien(theta1, theta4):
    """
    Calcule la matrice jacobienne du mécanisme en fonction des angles theta1 et theta4.
    
    Cette matrice permet de relier les vitesses angulaires des moteurs aux vitesses cartésiennes de la poignée. 
    
    Retourne :
        J : matrice 2x2 jacobienne
    """
    # Calculs intermédiaires (géométrie normalisée)
    D = (la + lb + lc / 2) / 3
    r1, r2 = la / D, lb / D
    s1, s4 = np.sin(theta1), np.sin(theta4)
    c1, c4 = np.cos(theta1), np.cos(theta4)

    # Variables d’orientation
    A = r1 * s1 - r1 * s4
    B = 2 * (lc / 2) / D + r1 * c1 + r1 * c4
    C = np.pi / 2 + np.arctan(A / B)

    # Dérivées géométriques
    D_val = 8 * r2 * np.sqrt(1 - (B**2 + A**2) / (4 * r2**2))
    E = r2 * np.sin(C) / (1 + A**2 / B**2) * np.sqrt(1 - (B**2 + A**2) / (4 * r2**2))
    Ep = r2 * np.cos(C) / (1 + A**2 / B**2) * np.sqrt(1 - (B**2 + A**2) / (4 * r2**2))

    # Éléments de la matrice jacobienne (dérivées partielles)
    J11 = -r1 * s1 / 2 - 2 * np.cos(C) * (A * r1 * c1 - B * r1 * s1) / D_val - E / B**2 * (B * r1 * c1 + A * r1 * s1)
    J12 = -r1 * s4 / 2 + 2 * np.cos(C) * (A * r1 * c4 + B * r1 * s4) / D_val - E / B**2 * (-B * r1 * c4 + A * r1 * s4)
    J21 = -r1 * c1 / 2 - 2 * np.sin(C) * (A * r1 * c1 - B * r1 * s1) / D_val + Ep / B**2 * (B * r1 * c1 + A * r1 * s1)
    J22 = -r1 * c4 / 2 + 2 * np.sin(C) * (A * r1 * c4 + B * r1 * s4) / D_val + Ep * B**2 * (-B * r1 * c4 + A * r1 * s4)

    return np.array([[J11, J12], [J21, J22]])

def dynamique(theta1, theta4, tau1, tau2):
    """
    Calcule la force exercée dans le plan cartésien (Fx, Fy) (N) à partir des couples moteurs (Nm).

    Utilise la transposée du jacobien pour transformer les efforts :
        τ = J^T · F  →  F = (J^T)^(-1) · τ

    Retourne :
        Fx, Fy : force résultante (N)
    """
    if tau1 != 0 or tau2 != 0:
        J_T = np.transpose(jacobien(theta1, theta4))
        F = np.linalg.solve(J_T, np.array([tau1, tau2]))
        return F[0], F[1]
    else:
        return 0, 0

def dynamique_inverse(theta1, theta4, Fx, Fy):
    """
    Calcule les couples moteurs nécessaires pour générer une force cartésienne donnée.

    Utilise l'équation inverse de la dynamique :
        τ = J^T · F

    Retourne :
        tau1, tau2 : couples à appliquer sur les moteurs (Nm)
    """
    tau = np.dot(np.transpose(jacobien(theta1, theta4)), np.array([Fx, Fy]))
    return tau[0], tau[1]

la, lb, lc = 0.28, 0.31, 0.08   # meters
th1_min_deg, th1_max_deg = 90, 170 # valeurs possible : de 90 à 170 degrès
th4_min_deg, th4_max_deg = 10, 90 #valeurs possible pour theta4 : de 10 à 90 degrès
step_deg = 1.0
front_only = True

th1_vals = np.deg2rad(np.arange(th1_min_deg, th1_max_deg + 1e-9, step_deg))
th4_vals = np.deg2rad(np.arange(th4_min_deg, th4_max_deg + 1e-9, step_deg))

xs, ys = [], []
for t1 in th1_vals:
    for t4 in th4_vals:
        p = cinematique(t1, t4)
        if p is None:
            continue
        x, y = p
        if (not front_only) or (y >= 0):
            xs.append(x); ys.append(y)

xs = np.array(xs); ys = np.array(ys)

fig, ax = plt.subplots(figsize=(7,7))
ax.scatter(xs, ys, s=1, alpha=0.5)
ax.plot([0, lc], [0, 0], marker='o', linestyle='--')
ax.set_aspect('equal', adjustable='box')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_title("Workspace (balayage de θ₁ et θ₄)\n"
             f"la={la:.2f} m, lb={lb:.2f} m, lc={lc:.2f} m")
plt.show()
