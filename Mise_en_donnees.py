#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on November 2025

@author: drapier


Tests pour les calculs statiques linéaires et charges critiques des EF d'Hermite 
=================================================================================

.. note::
    utilise la fonction d'affichage **fonct_Affichage(mot_clef, colormap, vp)** pas forcément adaptée, mais affichage à minima

"""
        
import numpy as np
import Hermitte_NL_Classes as Cl


# Initialisation de ce qui est de dimension connue
nb_elem = 0
nb_nodes = 0
    
nodes_global = []
elem_global = []
nodes_elem = []
   
# 1/ Maillage = 
# - définitions des noeuds = construire la liste nodes_global avec des objets noeuds; 
#     - peut être fait à partir d'un fichier texte contenant 3 colonnes : num_noeud, x, y'
# - définition des connectivités = donner une liste nodes_elem; 
#     - peut être fait à partir d'un fichier texte contenant 3 colonnes : num_elem, noeud1, noeud 2
#     - l'ordre des noeuds définit 'orientation : les charges réparties sont appliquées dans le repère LOCAL
    
longueur = 2.
Nl = 10
angle = 0.
# Fonction de Maillage d'une poutre droite de longueur donnée, avec Nl élements et formant un angle avec l'horizontale
# nodes_global, nodes_elem = Cl.fonct_Maille_Poutre(longueur, Nl, angle)
# nodes_global, nodes_elem = Cl.fonct_Maille_Poutre(longueur, Nl, 0)

# Création de la liste des noeuds - la position du noeud en partant de 0 est celle qui servira à stocker les ddls, indépendamment de son numéro
# 31/10/2025 : si chargements répartis définis dans le fichier des noeuds, on initialise les noeuds avec ces valeurs
nodes_global, nodes_elem = Cl.fonct_Maille_depuis_Fichier('Essai_noeuds.txt', 'Essai_elem.txt')
nb_elem = len(nodes_elem)
nb_nodes = len(nodes_global)

# Les propriétés et chargements répartis des éléments peuvent être précisés dans un np.tableau
E = 200.E3
rho = 2.6E-6
bs = 1000
hs = 100
g = 9.81E-3
ES = E * bs * hs
EI = E * bs * hs**3/12
Smasse = rho * bs * hs # masse de la section
meca_elem = np.repeat([[ES, EI, Smasse]],nb_elem,axis=0)

px = 0 #g * Smasse
py = 0
px_py = np.repeat([[px, py]],nb_elem,axis=0)

props_elem = [meca_elem, px_py]

# 2/ Créé un objet "problème éléments finis" à partir d'une liste de noeuds et des propriétés Meca_elem et px_py    
pb = Cl.pbEF_poutre(nodes_global, nodes_elem, props_elem)

# Créé le maillage correspondant aux connectivités et assemble les grandeurs (K_global, M_global, F_rep)
pb.Maillage_Assemblage()

# 3/ Conditions aux limites
total_dofs = pb.total_dofs

pb.CL_forces = []#[,1,-1/np.sqrt(2)],[11,2,-1./np.sqrt(2)]]#,[3,1,2E3]]
pb.CL_deplacements = [[101,1,0],[101,2,0],
                      [201,1,0],[201,3,0]]#,[8,3,0.4],[4,1,-0.4],[2,2,0.4]]

pb.Application_CL()


K_global, Kgeom_global, M_global = pb.K_global, pb.Kgeom_global, pb.M_global
F_con, F_rep, F_tot = pb.F_con, pb.F_rep, pb.F_tot

# 4/ Résolution 
dofs = pb.Resol()


# Mise à jour des champs cinématiques et statiques / pour pré-charges de flambage ou affichage
pb.Calcul_Def_Eff()

# 5/ Affichage - si pas de colormap donnée (coolwarm, bwr, seismic, ...), pas d'affichage de la colorbar 
pb.Affichage('M', 'coolwarm', vp = 5)

# reactions = pb.Calcul_Reactions()

# %%
 
# Calcul des charges de flambage

# A son initialisation,la rigidité géométrique n'est pas définie : les efforts normaux sont nuls avant le calcul des pré-contraintes
# -> le signe sera défini quand on aura initialisé avec l'état de contrainte calculé par le pb linéarisé
# -> le signe de la charge associée calculée changera en conséquence

# -> Après le calcul statique, on actualise Kgeom_global
# ATTENTION : Si Dirichlet non-homogène: pré-contraintes calculées à partir du pb statique => OK, MAIS
# Retrancher les contributions dues à la pénalité des CL Dirichlet dans K_global et F_tot, 

# pb.Application_CL(CL_forces, CL_deplacements, -1) # -1 <=> dans K_global, enlève les pénalités des CL Dirichlet non-homogènes
# pb.MAJ_Rigidite_Geom()

charge_min0 = []
mode0 = []
charge_min0, mode0 = pb.Calcul_Modes_Propres('Vibration', 3)

# Si on veut afficher les qtités associées aux modes propres - définis à une constant près !
# /!\ une fois ces quantités calculées pour affichage, les pré-contraintes deviennent celles associées au mode propre tracé
# pb.Calcul_Def_Eff()
# pb.Affichage('M', '', vp = 1)

# Exemple de mise à jour de propriétés d'éléments
for ind_elem, elem_temp in enumerate(pb.elem_global):
    Cl.Modif_Proprietes_Elem(pb, elem_temp.num_elem, ES, EI, Smasse*10000)
# # print(pb.elem_global[0])

# Calcul des vibrations propres  

# Exemple de création d'un élément puis calcul de vibration propre    
# Cl.Creation_Elem_Assemblage(pb,nb_elem+1, 5, 9, [1.E8, 1.E9,1.], [0,0])
# nb_elem += 1
charge_min1, mode1 = pb.Calcul_Modes_Propres('Vibration', 5)

# pb.Affichage('X', '', vp = 1)

# Cl.Creation_Elem_Assemblage(pb, 20, 5,9, [meca_elem], [px_py])

# %%
# # Tracé de la déformée, par exemple pour comparaison avec solution analytique
X = np.zeros(nb_nodes)
Y = np.zeros(nb_nodes)
# x1_values = np.zeros(Nl+1)
# x2_values = np.zeros(Nl+1)
x1_values = [node.x1 for node in nodes_global]
x2_values = [node.x2 for node in nodes_global]
for ind in range(nb_nodes):
    X[ind] = nodes_global[ind].x1 + dofs[ind].u1
    Y[ind] = nodes_global[ind].x2 + dofs[ind].u2

import matplotlib.pyplot as plt    
plt.plot(x1_values,x2_values,'x', color='b',label='initial')
plt.plot(X,Y,'o', color='r',label='deformee')
plt.legend()
plt.title('A changer')  
plt.show()


   
# Tests sur les matrices
# B = M_global
# est_symetrique = np.allclose(B, B.T)
# if est_symetrique:
#     print("La matrice Kgeom_global est symétrique.")
# else:
#     print("La matrice Kgeom_global n'est pas symétrique.")
# # Vérifier si B est définie positive en calculant sa décomposition de Cholesky
# try:
#     np.linalg.cholesky(B)
#     print("La matrice Kgeom_global est définie positive.")
# except np.linalg.LinAlgError:
#     print("La matrice Kgeom_global n'est pas définie positive.")




