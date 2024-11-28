#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Novembre 2024

@author: Sylvain Drapier


Classes et utilitaires pour résolution statique + charges de flambage + modes vibration avec des EF d'Hermitte en 2D 
====================================================================================================================
- éléments de poutre Hermitte en 2D (3 DDLs par noeud)
- calculs en statique y compris charges réparties
- calcul des charges critiques de flambage
- calcul des modes propres de vibration

- propriétés variables par élément
- ajout/retrait d'éléments
- affichage des résultats : u (dép),e (memb),k (courbure), N (eff normal), M (moment), 
    
:Partie 0: Initialisation des grandeurs physiques du problème

:Partie 1: **classes Node et Elem**

:Partie 2: Utilitaires : 
    
1. Discrétisation : avec fonctions/utilitaires ou depuis lecture fichiers noeuds et connectivités  
2. Création des éléments et assemblages    
3. Calcul des déformations/courbures e,K et des efforts/moments correspondants N et M
4. Trace le maillage déformé - poutres, noeuds, et chargements ponctuels + e, K, N, M, ES, EI, Smasse


:Partie 3: **classe pbEF_poutre** qui utilise toutes les autres définitions regroupée ici pour avoir des variables globales communes à toutes les fonctions
  

1/ Structures de données du problème
====================================
- **Classe Node** = Noeuds contenant :
    - numero
    - x et y : coordonnées
    - u et v, u1 et u2, theta: degrés de liberté dans les axes locaux et dans les axes structure
    - position : indice dans la liste 'nodes_global' des noeuds construite initialement
    
- **Classe Elem** = Eléments contenant :
    - numero
    - inode1 et inode2 : numéro des noeuds : ATTENTION l'orientation de l'élément dépend de l'ordre des noeuds
    - angle et longueur : caractéristiques géométriques
    - ES, EI, Smasse : caractéristiques physiques
    - px et py : chargements linéiques orientés dans le REPERE LOCAL de la poutre (orientation !)
    - grandeurs élémentaires (raideur initiale, raideur géométrique, masse, seconds membres)
    
    *Méthodes* = construction d'un élément avec calcul de ses grandeurs; mises à jour si changement propriétés et/ou pré-charge flambage, destruction

- **Classe pbEF_poutre** = = attributs et méthodes de tout le problème. Le passage des variables se fait par référence en interne à l'objet. Pas de variables globales.   
    *Structures de stockage* : 
        
    - elem_global = liste qui contient les éléments, 
    - nodes_global = liste qui contient les noeuds; elle définit l'ordre de stockage des ddls et donc des vecteurs et matrice, 
    - F_con, F_rep, F_tot = efforts nodaux
    - K_global, Kgeom_global, M_global = raideurs initiale, géométrique, et masse
    
    
2/ Variables globales qui sont rattachées à l'objet principal de type *pbEF_poutre*
===================================================================================
.. attribute:: nodes_global : list([])

    Liste des noeuds 
.. attribute:: nodes_elem : list([[,,]])

    Liste d'éléments qui constituent le maillage: numéro d'élément et paires de noeuds (l'ordre des noeuds donne l'orientation)     
.. attribute:: K_global, Kgeom_globaln M_global : np.array((,))

    Matrices de ridité élastiques et géométrique, et de masse 
.. attribute:: F_con, F_rep, F_tot : np.array()

    Efforts concentrés, répartis, et totaux 
.. attribute:: elem_global : list([])

    Liste des éléments 
.. attribute:: meca_elem : list(np.array(3))

    Liste des propriétés mécaniques; on peut s'en passer 
    
.. attribute:: (px, py) : 2 x float

    Chargements répartis : charges linéiques [N.m-1] orientées dans le repère des poutres
    
.. attribute:: fixed_dofs : list([])

    Liste des noeuds bloqués  
            
    
::
       
    # Définition des propriétés par élément : les données matériaux et les chargements répartis sont 
    # stockés dans un tableau **props_elem** de type np.array à 3 dimensions
    # Unités SI (mksA)

    # Par exemple pour toutes les propriétés identiques
    E = 70.E9
    rho = 2.6
    bs = 0.15
    hs = 0.3
    ES = E * bs * hs
    EI = E * bs * hs**3/12
    Smasse = rho * bs * hs # masse de la section
    meca_elem = np.repeat([[ES, EI,Smasse]],nb_elem,axis=0)
    
    px = 0.
    py = -0.
    px_py = np.repeat([[px, py]],nb_elem,axis=0)
    
    props_elem = [meca_elem, px_py]
      
    


3/ Classes et méthodes
========================

"""


"""
=================================================================================

                            PARTIE 0 : Initialisation

=================================================================================

"""

       
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import ticker
import sys



"""
=================================================================================

                        PARTIE 1 : classes Node et Elem

=================================================================================
"""

"""Définition de la classe Node : numéro, coordonnées, inconnues cinématiques, position dans la liste"""
class Node:
    """ Classe Node
    
    .. attribute:: numero : int
    .. attribute:: x1, x2 : (float,float) 
    
        coordonnées
    .. attribute:: u, v : (float,float) 
    
        déplacements dans le repère de l'élément
    .. attribute:: u1, u2 : (float,float) 
    
        déplacements dans le repère de structure
    .. attribute:: theta : float 
    
        ddl rotation    

    .. attribute:: position : int 
    
        position du noeud dans la liste des noeuds - utile pour accès aux ddls         
    """
    def __init__(self, numero, position, x1, x2, u=0, v=0, u1=0, u2=0, theta=0):
        self.numero = numero
        self.position = position
        self.x1 = x1
        self.x2 = x2
        self.u = u # dans le repère local       
        self.v = v 
        self.u1 = u1
        self.u2 = u2      
        self.theta = theta

    def __str__(self):
        return f"Noeud {self.numero}, x1 = {self.x1:2f}, x2 = {self.x2:2f}, \n u = {self.u:2f}, v = {self.v:2f}, \n u1 = {self.u1:2f}, u2 = {self.u2:2f}, \n theta = {self.theta} \n position = {self.position}"


def Renvoi_index_node(n_node, liste):
    """ renvoi l'indice du noeud n_node dans la liste de noeuds 'liste' """
    for i in range(len(liste)):
        if liste[i].numero == n_node:
            return i  # Renvoie la position de l'élément si trouvé
    return -1  # Renvoie -1 si l'élément n'est pas trouvé 

"""
======================================  utilitaires de la classe Elem ===============================
"""
def calcul_kel(alpha, beta, C, S, le):
    """ Calcul la matrice de rigidité élastique élémentaire dans le repère de structure.
    
- alpha, beta : rigidités de tension et de flexion 
- C, S : cosinus et sinus de l'angle
- le : longueur de l'élément

.. attribute:: kel : np.array((6,6))
    
    """
    kel_elem = np.array([
        [C**2*alpha+12*S**2*beta, C*S*(alpha-12*beta), -6*S*beta*le, -C**2*alpha-12*S**2*beta, -C*S*(alpha-12*beta), -6*S*beta*le],
        [C*S*(alpha-12*beta), S**2*alpha+12*C**2*beta, 6*C*beta*le, C*S*(-alpha+12*beta), -S**2*alpha-12*C**2*beta, 6*C*beta*le],
        [-6*S*beta*le, 6*C*beta*le, 4 * beta*le**2, 6*S*beta*le, -6*C*beta*le, 2 * beta *le**2],
        [-C**2*alpha-12*S**2*beta, C*S*(-alpha+12*beta), 6*S*beta*le, C**2*alpha+12*S**2*beta, C*S*(alpha-12*beta), 6*S*beta*le],
        [-C*S*(alpha-12*beta), -S**2*alpha-12*C**2*beta, -6*C*beta*le, C*S*(alpha-12*beta), S**2*alpha+12*C**2*beta, -6 *C * beta * le],
        [-6*S*beta*le, 6*C*beta*le, 2 * beta *le**2, 6*S*beta*le, -6*C*beta*le, 4 * beta*le**2]
        ])
    return kel_elem
    
def calcul_fe(C, S, le, px, py ):
    """ Second membre des efforts répartis recalculé avec changement de base """
    return (le/2) * np.array([px*C-py*S, -px*S+py*C, py*le/6, px*C-py*S, -px*S+py*C, -py*le/6])

def calcul_kg(C, S, le, Effort_N):
    """ Calcul la matrice de rigidité géométrique élémentaire dans le repère de structure et met à jour avec l'effort normal sur les composantes 1,2 et 4,5.
    
- C, S : cosinus et sinus de l'angle
- le : longueur de l'élément
- Effort_N : effort normal dans l'élément exprimé dans le repère local - cf démonstrations dans la documentation

.. attribute:: kg : np.array((6,6))
    
    """

    kg_elem = Effort_N/(30 * le) * np.array([
        [36*S**2, -36*C*S, -3*S*le, -36*S**2, 36*C*S, -3*S*le],
        [-36*C*S, 36*C**2, 3*C*le, 36*C*S, -36*C**2, 3*C*le],
        [-3*S*le, 3*C*le, 4*le**2, 3*S*le, -3*C*le, -le**2],
        [-36*S**2, 36*C*S, 3*S*le, 36*S**2, -36*C*S, 3*S*le],
        [36*C*S, -36*C**2, -3*C*le, -36*C*S, 36*C**2, -3*C* le],
        [-3*S*le, 3*C*le, -le**2, 3*S*le, -3*C*le, 4*le**2]
        ])
    return kg_elem

def calcul_M(Smasse, C, S, le):
    """ Calcul la matrice de masse dans le repère de structure.
    
- C, S : cosinus et sinus de l'angle
- le : longueur de l'élément
- Smasse : masse par unité de longueurs l'élément exprimé dans le repère local

.. attribute:: Mel : np.array((6,6))
    
    """
    M_elem = Smasse*le/420 * np.array([
        [140*C**2 + 156*S**2, -16*C*S, -22*le*S, 70*C**2 + 54*S**2, 16*C*S, 13*le*S],
        [-16*C*S, 156*C**2 + 140*S**2, 22*C*le, 16*C*S, 54*C**2 + 70*S**2, -13*C*le],
        [-22*le*S, 22*C*le, 4*le**2, -13*le*S, 13*C*le, -3*le**2],
        [70*C**2 + 54*S**2, 16*C*S, -13*le*S, 140*C**2 + 156*S**2, -16*C*S, 22*le*S],
        [16*C*S, 54*C**2 + 70*S**2, 13*C*le, -16*C*S, 156*C**2 + 140*S**2, -22*C*le],
        [13*le*S, -13*C*le, -3*le, 22*le*S, -22*C*le, 4*le**2]
        ])
    return M_elem

class Elem:
    '''
    Classe Elément.

    Les noeuds (de type Node) inode1 et inode2 existent et sont utilisés

    .. attribute:: num_elem : int
    .. attribute:: inode1, inode 2 : int
    
        les noeuds sont initialisés avec l'instance de l'élément        
    .. attribute:: meca_elem : np.array(3)
    
        contient dans l'ordre ES, EI, Smasse (masse section)
    .. attribute:: px_py : np.array(2) 
    
        efforts linéiques longitudinal et transverse   
    .. attribute:: nodes_global : list 
    
        list contenant les noeuds du problème global (création des noeuds)        
    .. attribute:: angle : float [= 0]
    
        angle de l'élément par rapport à l'axe x_1
    .. attribute:: le : float [= 0]
    
        longueur élémentaire   
    .. attribute:: kel_e : float 
    
        rigidité élastique élémentaire  
    .. attribute:: kg_e : np.array((6,6)) [=[]]
    
        rigidité géométrique élémentaire   
    .. attribute:: fe : np.array(6) [=[]]
    
        second membre élémentaire   
    .. attribute:: def_eff : np.array(4) [=[]]
    
        contient les déformations et les efforts de l'élément dans le repère local : [u',v'',N,M]   
       
    ''' 
    def __init__(self, num_elem, inode1, inode2, meca_elem, px_py, nodes_global, angle=0,  le=0, kel_e=[], kg_e=[], f_e=[], M_e=[], def_eff=[]):     
        self.num_elem = num_elem
        ES = self.ES = meca_elem[0]
        EI = self.EI = meca_elem[1]
        self.Smasse = meca_elem[2]
        self.px = px_py[0]
        self.py = px_py[1]
      
        self.angle = angle  
        self.le = le
        self.kel_e = kel_e
        self.kg_e = kg_e
        self.f_e = f_e
        self.M_e = M_e        
        self.def_eff = np.zeros(4)  # e, K, N, M      
 
        # Récupère les 2 objets Node dont les numéros sont connus
        self.node1 = nodes_global[Renvoi_index_node(inode1, nodes_global)]
        self.node2 = nodes_global[Renvoi_index_node(inode2, nodes_global)]
        
        # Calculer la longueur et l'angle des éléments connaissant les noeuds
        x1_1 = self.node1.x1
        x1_2 = self.node1.x2
        x2_1 = self.node2.x1
        x2_2 = self.node2.x2       
        le = self.le = np.sqrt( (x2_1 - x1_1)**2 + (x2_2 - x1_2)**2 )

        
        # Précautions à prendre sur l'angle, il définit l'orientation de la poutre
        #      |\     n
        #        \  n   
        #         +    ^ x2
        #          t   |
        #           t  | 
        #            t |
        #             \|
        #              +-----------> x1
        # global C, S
  
        # Ne pas regarder le signe de l'angle, le laisser défini par l'ordre des noeuds
        # (self.C, self.S) = (1., 0.)     
        # if np.abs(x2_2 - x1_2) > 1E-13 : # angle par rapport à l'horizontal x1
        #     if np.abs(x2_1 - x1_1) > 1E-13:
        #         self.C = ( (x2_1 - x1_1) / le)
        #         self.S = ( (x2_2 - x1_2) / le )                
        #         self.angle = np.arctan( self.S / self.C )
        #         if ((self.C*self.S > 0) and (self.C<0)):
        #             self.angle += np.pi
        #         elif ((self.C*self.S <0) and (self.C<0)):
        #             self.angle += np.pi
        #     else :
        #         (self.C, self.S) = (0, 1)
        #         self.angle = np.pi/2
                
        (self.C, self.S) = (1., 0.)     
        if np.abs(x2_2 - x1_2) < 1E-13 : # aligne sur x1
            self.C = 1 * np.sign(x2_1 - x1_1)
            if x2_1 > x1_1 :
                self.angle = 0
            else :
                self.angle = np.pi
#np.sign renvoie 0 si qtité nulle
        elif np.abs(x2_1 - x1_1) < 1E-13: # aligne sur x2
            (self.C, self.S) = (0, 1 * np.sign(x2_2 - x1_2))
            self.angle = np.pi/2 * np.sign(x2_2 - x1_2)
        else :
            self.C = ( (x2_1 - x1_1) / le)
            self.S = ( (x2_2 - x1_2) / le )                
            self.angle = np.arctan( self.S / self.C )
            if ((self.C*self.S > 0) and (self.C<0)):
                self.angle += np.pi
            elif ((self.C*self.S <0) and (self.C<0)):
                self.angle += np.pi
                
        alpha = ES / le
        beta = EI / le**3
        
        self.kel_e = calcul_kel(alpha, beta, self.C, self.S, le )
        self.f_e = calcul_fe(self.C, self.S, le, self.px, self.py )
        self.kg_e = calcul_kg(self.C, self.S, le, self.def_eff[2])
        self.M_e = calcul_M(self.Smasse,self.C, self.S, le)

    def mise_a_jour_kel(self):
        """ Si propriétés changent (mécanique, angle, longueur)"""
        alpha = self.ES / self.le
        beta = self.EI / self.le**3
        self.kel_e = calcul_kel(alpha, beta, self.C, self.S, self.le)

    def mise_a_jour_fe(self):
        """ Si propriétés changent(angle, longueur, chargement)"""
        self.f_e = calcul_fe(self.C, self.S, self.le, self.px, self.py )
        
    def mise_a_jour_kg(self):
        """ Si propriétés et/ou effort normal changent (angle, longueur, effort)"""
        self.kg_e = calcul_kg(self.C, self.S, self.le, self.def_eff[2] )

    def mise_a_jour_Me(self):
        """ Si propriétés changent (masse linéique, angle, longueur)"""
        self.M_e = calcul_M(self.Smasse, self.C, self.S, self.le)

        
    def __str__(self):
        return f"Elément{self.num_elem} \n {self.node1} \n {self.node2} \n \n angle = {self.angle*180/np.pi} degrès, \n longueur = {self.le}, ES = {self.ES}, EI = {self.EI}, Smasse = {self.Smasse}, \n px = {self.px}, py = {self.py}, "


"""
=================================================================================

                        PARTIE 2 : Utilitaires

=================================================================================

    1/ Détermine les altitudes des noeuds situés à une abscisse donnée
    
    2/ Création des éléments et assemblages
    
    2'/ Mise à jour des rigidités des éléments si ES et/ou EI changent
    
    3/ Calcule des déformations/courbures r,K et des efforts/moments correspondants N et M
    
    4/ Trace le maillage déformé - poutres, noeuds, et chargements ponctuels + e, K, N, M, ES, EI, Smasse
    
"""
""" 
1/
Calcul des caracteristiques  géométriques des courbes décrivant les peaux
- cercle qui passe par les points (0,0),(l,h),(2l,0):
- Polynômes de fit  
"""
def y_cercle(x1,x1_c,x2_c,R2):
    """ Calcul des caracteristiques  géométriques des courbes décrivant les peaux
    - cercle qui passe par les points (0,0),(l,h),(2l,0):    x1_c = l, x2_c = (h^2-l^2)/2h et R^2 = (h^2+l^2/2h)^2
    Pour une abscisse x1, on calcule la plus grande racine x2_max : (x2 - x2c)^2 - R^2 - (x1 - x1_c)^2 =0 => x2^2 - 2 x2 x2_c + x2_c^2 - (R^2 + (x1 - x1_c)^2)
    """
    rac_delta = np.sqrt(R2-(x1-x1_c)**2)
    return x2_c + rac_delta  

def y_inf(xi):
    return (-0.00011075285292718527*xi**7+0.002068379319946006*xi**6+-0.015056981533807745*xi**5 
        + 0.05141645527951446*xi**4+-0.07847883297149826*xi**3 +0.07666880082908949*xi**2 + -0.018798143630654435*xi
        -0.8637973132408278)
    
def y_sup(xi):
    return (0.0003557519035056747*xi**7 - 0.0064267583698702745*xi**6 + 0.044158679925716184*xi**5 
            - 0.13889850173484505*xi**4 + 0.18553509121476225*xi**3 - 0.09162182118970996*xi**2 + 
            0.025386206141218705*xi + 0.6771458904146 )

def Renvoi_index_elem(n_elem, liste):
    """ renvoi l'indice dans la liste elem_global d'un élément de numéro n_elem """
    for i in range(len(liste)):
        if liste[i].num_elem == n_elem:
            return i  # Renvoie la position de l'élément si trouvé
    return -1  # Renvoie -1 si l'élément n'est pas trouvé 

"""
2/ 
- Initialisation des noeuds et des connectivités depuis des fichiers textes
- Création des éléments dans elem_global et assemblage des rigidités et efforts répartiss 
    entrée : numéro d'élément, noeud1, noeud 2, liste éléments globale, propriétés méca, charge répartie, liste globale de noeuds
    sortie : efforts répartis et rigidités élastique et géométrique
"""

def fonct_Maille_depuis_Fichier(nomfic_noeuds, nomfic_elements):
    """
Initialise les listes de données pour une sructure réticulée à partir de fichiers contenant les noeuds (3 colonnes : num_noeud, x, y) et 
les connectivités (3 colonnes : num_elem, noeud1, noeud2)

*Retourne :* la liste des noeuds 'nodes_global' et la connectivité des éléments 'nodes_elem'

** Vérifications ** : 
    - 1 tous les noeuds nécessaires pour construire les éléments existent; sinon arrêt et affichage Erreur
    - 2 les noeuds non utilisés sont supprimés - message d'avertissement
    """
    
    nodes_global=[]
    ipos_node = 0
    liste_nodes_temp = []
    try:
        with open(nomfic_noeuds, 'r') as fichier:
            for ligne in fichier:
                if ligne.strip() and not ligne.strip().startswith('#'):
                    ipos_node += 1
                    valeurs = ligne.strip().split(',')
                    if len(valeurs) != 3:
                        raise ValueError("Ligne",ligne," mal formatée")
                    else : 
                        nodes_global.append(Node(int(valeurs[0]), ipos_node-1, float(valeurs[1]), float(valeurs[2])))    
                        liste_nodes_temp.append(int(valeurs[0]))
    
    except ValueError as e:
        print(f"Erreur dans le fichier : {e}")
    except FileNotFoundError:
        print("Le fichier est introuvable.")
       
    nodes_elem = []
    inode1 = 0
    inode2 = 0  
    liste_node_elem_temp = []
    try:
        with open(nomfic_elements, 'r') as fichier:
            for ligne in fichier:
                if ligne.strip() and not ligne.strip().startswith('#'):
                    valeurs = ligne.strip().split(',')
                    (inode1, inode2) = ( int(valeurs[1]) , int(valeurs[2]) )
                    # vérifie que les noeuds nécessaires sont bien définis
                    if len(valeurs) != 3:
                        raise ValueError("Ligne",ligne," mal formatée")
                    elif inode1 in liste_nodes_temp :
                        if inode2 in liste_nodes_temp :
                            nodes_elem.append([int(valeurs[0]), inode1, inode2])
                            liste_node_elem_temp.append(inode1)
                            liste_node_elem_temp.append(inode2)
                    else :
                        raise ValueError("L'élément",int(valeurs[0])," utilise un noeud qui n'est pas défini")
    
    except ValueError as e:
        print(f"Erreur dans le fichier : {e}")
    except FileNotFoundError:
        print("Le fichier est introuvable.")  
        
    # vérifie que tous les noeuds sont bien utilisés 
    result = [x for x in liste_nodes_temp if x not in liste_node_elem_temp]
    if not result == [] :
        raise ValueError("Le(s) noeud(s)", result, "n(e sont) pas utilisé(s)")

    
    
    return nodes_global, nodes_elem

def Creation_Elem_Assemblage(pbEF, num_elem, inode1, inode2, pptes_meca, px_py):

    """ 
Création d'un élement et intégration de sa raideur élastique, sa matrice de masse, et efforts dans les grandeurs globales, en ne prenant en compte les contritions correspondant aux CL Dirichlet (permet d'ajouter des éléments a posteriori)
* Remarque :* La rigidité géométrique est initialisée à 0, elle sera mise à jour après le calcul des pré-contraintes dans MAJ_Rigidite_geom_Elem(num_elem)
*Parameters*

    num_elem : int
        numéro de l'élément à créer
    inode1 : int
        noeud 1
    inode2 : int
        noeud 2
    elem_global : list
        liste globale des éléments
    pptes_meca : np.array(3)
        EI, ES, et Smasse
    px, py : float
        Charges linéiques longitudinale et transverse dans le repère de l'élément (orientation donnée par l'ordre des noeuds)

*Returns*

    None.

    """
     
    # global fixed_dofs

    fixed_dofs = pbEF.fixed_dofs
    elem_global = pbEF.elem_global
    nodes_global = pbEF.nodes_global 
    F_rep = pbEF.F_rep
    K_global = pbEF.K_global    
    M_global = pbEF.M_global 
    
    elem_global.append( Elem(num_elem, inode1, inode2, pptes_meca, px_py, nodes_global) )        
    elem_temp = elem_global[Renvoi_index_elem(num_elem, elem_global)]
    # on travaille sur les ddls => position des noeuds dans la liste construite
    ipos_node1 = elem_temp.node1.position
    ipos_node2 = elem_temp.node2.position
    dof_global = [3*ipos_node1, 3*ipos_node1 + 1, 3*ipos_node1+2, 3*ipos_node2, 3*ipos_node2 +1 , 3*ipos_node2 + 2]
    dof_local = []
    # ne pas prendre en compte les ddls dans les CL Dirichlet : on ne touche pas aux contributions correspondantes
    for i in range(6):
        if (dof_global[i] not in fixed_dofs):
            dof_local.append(i)
  
    for i in dof_local:
        F_rep[dof_global[i]] += elem_temp.f_e[i]
        for j in dof_local:
            K_global[dof_global[i], dof_global[j]] += elem_temp.kel_e[i, j]
            M_global[dof_global[i], dof_global[j]] += elem_temp.M_e[i, j]

    return F_rep, K_global, M_global    
    

def Modif_Proprietes_Elem (pbEF, num_elem, ES_el, EI_el, Smasse_el):
    """
Met à jour la rigidité élastique GLOBALE quand les 
propriétés méca ES, EI ou Smasse d'un élément changent
*Remarque 1 :* On ne met pas à jour la rigidité géométrique qui sera mise à jour de toute façon quand les pré-contraintes auront été recalculées - cf fonction MAJ_rigidite_geometrique
*Remarque 2 :* Précaution à prendre, avant de retrancher les contributions puis de les remplacer, s'assurer qu'elles ne correspondent pas à des ddls dans les CL de Dirichlet    

*Parameters*

    num_elem : int
        
    ES_el : float
        
    EI_el : float
    
    S_masse : float
        

*Returns*

    None.

    """
    
    K_global = pbEF.K_global
    M_global = pbEF.M_global
    fixed_dofs = pbEF.fixed_dofs
    elem_global = pbEF.elem_global
    
    elem_temp = elem_global[Renvoi_index_elem(num_elem, elem_global)]
    elem_temp.ES = ES_el
    elem_temp.EI = EI_el
    elem_temp.Smasse = Smasse_el
    ipos_node1 = elem_temp.node1.position
    ipos_node2 = elem_temp.node2.position
    dof_global = [3*ipos_node1, 3*ipos_node1 + 1, 3*ipos_node1+2, 3*ipos_node2, 3*ipos_node2 +1 , 3*ipos_node2 + 2]
    dof_local = []
    # ne pas prendre en compte les ddls dans les CL Dirichlet : on ne touche pas aux contributions correspondantes (déjà mises à 1 ou 0)
    for i in range(6):
        if (dof_global[i] not in fixed_dofs):
            dof_local.append(i)
            
    # print('dof_global = ',dof_global)
    # print('fixed_dofs =', fixed_dofs)
    # print('dof_local = ',dof_local) 

    # il faut d'abord supprimer les contributions de l'élément à modifier
    for i in dof_local:
        for j in dof_local:
            K_global[dof_global[i], dof_global[j]] -= elem_temp.kel_e[i, j]
            M_global[dof_global[i], dof_global[j]] -= elem_temp.M_e[i, j]
            # print('K_global[',dof_global[i],',',dof_global[j],'] -= elem_temp.kel_e[',i,',',j,']')
    
    elem_temp.mise_a_jour_kel()
    elem_temp.mise_a_jour_Me()
    for i in dof_local:
        for j in dof_local:
            K_global[dof_global[i], dof_global[j]] += elem_temp.kel_e[i, j]
            M_global[dof_global[i], dof_global[j]] += elem_temp.M_e[i, j]           

def fonct_MAJ_rigidite_geom_elem(pbEF, num_elem):
    """
Met à jour la rigidité géométrique globale pour l'élément *num_elem* pour le calcul des charges critiques : 
    retranche puis remplace ses composantes dans la rigidité globale par Effort_N_(repère-local)*kg_e[,] et met à jours les CL

*Méthode :* On retranche les contributions, avant d'ajouter les nouvelles contributions    

*Remarque :* Précaution à prendre, avant de retrancher les contributions puis de les remplacer, s'assurer qu'elles ne correspondent pas à des ddls dans les CL de Dirichlet (contributions nulles dans ce cas)    

*Parameters*

    num_elem : int

*Returns*
   
    None.

    """

    Kgeom_global = pbEF.Kgeom_global
    fixed_dofs = pbEF.fixed_dofs
    elem_global = pbEF.elem_global
    
    elem_temp = elem_global[Renvoi_index_elem(num_elem, elem_global)]
    ipos_node1 = elem_temp.node1.position
    ipos_node2 = elem_temp.node2.position
    dof_global = [3*ipos_node1, 3*ipos_node1 + 1, 3*ipos_node1+2, 3*ipos_node2, 3*ipos_node2 +1 , 3*ipos_node2 + 2]
    dof_local = []
    # avant de retrancher les contributions : vérifier si les ddls impliqués ne correspondent pas à une CL de Dirichlet
    for i in range(6):
        if (dof_global[i] not in fixed_dofs):
            dof_local.append(i)

    for i in dof_local:
        for j in dof_local:
            Kgeom_global[dof_global[i], dof_global[j]] -= elem_temp.kg_e[i, j]
    
    elem_temp.mise_a_jour_kg()
    for i in dof_local:
        for j in dof_local:
            Kgeom_global[dof_global[i], dof_global[j]] += elem_temp.kg_e[i, j]

    # Finalement, plus besoin de mettre à jour les composantes concernées par la CL Dirichlet, elles n'ont pas été changées.
    # for dof in fixed_dofs:
    #      Kgeom_global[dof, :] = 0
    #      Kgeom_global[:, dof] = 0
    #      # proposition de https://bleyerj.github.io/comet-fenicsx/tours/eigenvalue_problems/buckling_3d_solid/buckling_3d_solid.html
    #      # si on met 1 pour les CL Dirichlet -> vp =1, polluent le spectre; si on met 0 -> infty
    #      Kgeom_global[dof, dof] = 0            
              
    return  


def Detruit_Elem(pbEF, n_elem):
    """
Supprime les contributions de l'élément n_elem et le supprime de la liste globale des éléments

*Parameters*
    
    n_elem : int
        

*Returns*
   
    None.

    """
    elem_global = pbEF.elem_global

    ind_destroy = Renvoi_index_elem(n_elem, elem_global)
    Modif_Proprietes_Elem (n_elem, 0, 0, 0, pbEF )
    del elem_global[ind_destroy]

    return 


def fonct_Calcul_Modes_Propres(pbEF, mot_clef, n_modes,):
    """
    Calcul des 'n_modes' pulsations et modes propres
    * Variable mot_clef :* prend 2 valeurs 'Flambage' ou 'Vibration' 
    * Returns :* n_modes premières valeurs propres et vecteurs propres attachés à pbEF
    """    

    eigenvects=[]
    eigenvals=[]
    pbEF.vibration = False # pour les affichages ensuite
    
    if mot_clef == 'Flambage' :
        mat_B = - pbEF.Kgeom_global
    else :
        mat_B = pbEF.M_global
        pbEF.vibration = True
    eigenvals, eigenvects = sc.linalg.eig( pbEF.K_global, mat_B,left=False, right=True )
    
    # Étape 1 : Filtrer les valeurs propres intéressantes, i.e. celles qui ne sont pas infinity (rappel: CL=0)
    ind_eig_interet = np.where(~np.isinf(eigenvals))[0] # tous les indices d'intérêt
    eig = eigenvals[ind_eig_interet]
    abs_eig = np.abs(eig) 
     
    # Étape 2 : Trouver les 'n_modes' plus petites valeurs propres
    indices_sorted = np.argsort(abs_eig)  # Indices triés des valeurs propres filtrées
    indices_min = indices_sorted[:n_modes]  # Prendre les indices des 'n_modes' plus petites valeurs
    indices_min_eigenvals = ind_eig_interet[indices_min] # Associer indices des valeurs propres mini retenues à leur position dans le vecteur initial 
       
    # Étape 3 : Récupérer les valeurs d'intérêt 
    pbEF.mode_min = np.real(eigenvects[:, indices_min_eigenvals])
    pbEF.vp_min = np.real(eig[indices_min])
    
    if mot_clef == 'Vibration': 
        pbEF.vp_min = np.sqrt(pbEF.vp_min)  # Les 'n_modes' plus petites pulsations propres = racines(omega^2)
    
    return pbEF.vp_min, pbEF.mode_min


def fonct_Maillage_Assemblage(pbEF):

    """
Création des éléments et assemblage des rigidités, matrice de masse, et des efforts répartis

    """

    meca_elem = pbEF.props_elem[0]
    px_py =  pbEF.props_elem[1]
    nodes_elem = pbEF.nodes_elem
    
    if len(nodes_elem) :
        for indice, entier in enumerate(nodes_elem):
            Creation_Elem_Assemblage(pbEF, entier[0], entier[1], entier[2], meca_elem[indice], px_py[indice])    
            
    return



def fonct_Application_CL(pbEF, forces, deplacements):
    """ Prise en compte des conditions de Neumann et Dirichlet homogènes  """

    # on vérifie que les noeuds où sont imposées les CL existent bien; sinon Erreur
    liste_node_temp = [sous_liste.numero for sous_liste in pbEF.nodes_global]
    liste_node_forces = []
    liste_node_dep = []
    if len(forces) != 0 : # on peut ne pas avoir d'efforts, pour les vibrations par exemple
        liste_node_forces = [forces_temp[0] for forces_temp in forces]
    liste_node_dep = [dep_temp[0] for dep_temp in deplacements]
    if not ( all(x in liste_node_temp for x in liste_node_dep) and (all(x in liste_node_temp for x in liste_node_forces) ) ) :
            sys.exit("Une condition aux limites est imposée sur un noeud qui n'existe pas.")  
                         
    indices = [Renvoi_index_node(row[0], pbEF.nodes_global) * 3 + (row[1] - 1) for row in forces]
    values = [row[2] for row in forces]
    # if not force[0:] in 
    pbEF.F_con[indices] = values
    # print(indices, values)
    pbEF.F_tot = pbEF.F_con + pbEF.F_rep
    
    pbEF.fixed_dofs = [Renvoi_index_node(row[0],pbEF.nodes_global)*3+(row[1]-1) for row in deplacements]

    """ Prise en compte des conditions Dirichlet homogènes  """
    # pour les inconnues bloquées : diagonale unitaire pour rigidité, et diagonale nulle pour masse
    for dof in pbEF.fixed_dofs:
         # pbEF.fixed_dofs = fixed_dofs
        pbEF.K_global[dof, :] = 0
        pbEF.K_global[:, dof] = 0
        pbEF.K_global[dof, dof] = 1
        
        pbEF.M_global[dof, :] = 0
        pbEF.M_global[:, dof] = 0
        pbEF.M_global[dof, dof] = 0
        
         # proposition de https://bleyerj.github.io/comet-fenicsx/tours/eigenvalue_problems/buckling_3d_solid/buckling_3d_solid.html
         # si on met 1 pour les CL Dirichlet -> vp =1, polluent le spectre; si on met 0 -> infty
         # 
         # comme la mise à jour de cette rigidité ne modifie pas les contributions liées aux CL Dirichlet, 
         # et que K_geom_global est initialisé à 0, on ne touche rien
         # pbEF.Kgeom_global[dof, :] = 0
         # pbEF.Kgeom_global[:, dof] = 0
         # pbEF.Kgeom_global[dof, dof] = 0
        
        pbEF.F_tot[dof] = 0
     

## ======================================== Fonctions pour tester les poutres  ========================

def fonct_Maille_Poutre(lp, Nl , angle):
    """
Maille une poutre de longueur lp avec Nl éléments, formant éventuellement un angle par rapport à x_1 
*Retourne :* la liste des noeuds et la liste des connectivités initialisées
    """
    
    nodes_global = []
    nodes_elem = []
    
    Dl = lp / Nl
    for i in range(Nl+1):
        xn = i*Dl*np.cos(angle)
        yn = i*Dl*np.sin(angle)
        nodes_global.append(Node(i+1,i,xn,yn))

    for i in range(Nl):        
        nodes_elem.append([i+1,i+1,i+2])

    return nodes_global, nodes_elem
      

""" 
3/
    Post-traitement : Calculs dans le repère local de l'élément, puis déformations et efforts internes
""" 
def fonct_Calcul_Def_Eff(pbEF):
    """
Pour tous les éléments, calcul les déformations et contraintes généralisées par élément dans le repère local - supposé constant au centre de l'élément (courbure) 
* Remarque :* Nécessite d'avoir mis à jour les déplacements aux noeuds - dans pb.Resol()
*Returns*
    
    elem_global : list
        liste globale des éléments

    """
    elem_global = pbEF.elem_global
        
    ind_elem = 0
    for ind_elem in range(len(elem_global)):
        # print("ind_elem {}".format(ind_elem))

        Elem_temp = elem_global[ind_elem]
        # inode1 = Elem_temp.node1.numero
        # inode2 = Elem_temp.node2.numero
        le = Elem_temp.le
        
        (C, S) = (np.cos(Elem_temp.angle), np.sin(Elem_temp.angle))
              
        # déplacements (u,v) aux noeuds dans le repère local : u=u1C+u2S, v=-u1S+u2C
        elem_global[ind_elem].node1.u = Elem_temp.node1.u1*C + Elem_temp.node1.u2*S
        elem_global[ind_elem].node1.v = -Elem_temp.node1.u1*S + Elem_temp.node1.u2*C
        elem_global[ind_elem].node2.u = Elem_temp.node2.u1*C + Elem_temp.node2.u2*S
        elem_global[ind_elem].node2.v = -Elem_temp.node2.u1*S + Elem_temp.node2.u2*C
        
        # Evaluation des déformations et efforts au centre de l'élément, en (x1+x2)/2
        # en tension, la déformation est simplement la différence (1/le)*(u_2-u_1) en parcourant les x croissants
        # alpha1 = 1/2 # au centre de l'élément (x-x1=1/2 le)
        # alpha2 = -alpha1          
        # en flexion, calcul des courbures à partir des interpolations
        # elem_global[ind_elem].def_eff[1] = Npp[0]*Elem_temp.node1.v + Npp[1]*Elem_temp.node1.theta + Npp[2]*Elem_temp.node2.v + Npp[3]*Elem_temp.node2.theta  
        # on peut encore simplifier les expressions car au centre de l'élément : v"=(1/le)*(theta2-theta1)      
        # Npp[0] = (6/le**2)*(-1 +2*alpha1 )  # au centre : 0
        # Npp[1] = (2/le)*(-2 + 3*alpha1 )    # au centre : -1/le
        # Npp[2] = -(6/le**2)*(1 + 2*alpha2 ) # au centre : 0
        # Npp[3] = (2/le)*(2 + 3*alpha2 )     # au centre : 1/le      
        elem_global[ind_elem].def_eff[0] = (1/le)*( Elem_temp.node2.u - Elem_temp.node1.u )
        elem_global[ind_elem].def_eff[1] = (1/le)*( Elem_temp.node2.theta - Elem_temp.node1.theta )  
        elem_global[ind_elem].def_eff[2] = elem_global[ind_elem].def_eff[0] * Elem_temp.ES
        elem_global[ind_elem].def_eff[3] = elem_global[ind_elem].def_eff[1] * Elem_temp.EI
        # print("EffInt[",ind_elem,"]", Eff_int[ind_elem])
        
    return  


""" 
4/
Affichages : maillage + sa déformée, et les efforts imposés

On raisonne par élément : affichage à partir des informations des éléments tous parcourus, 
pour ne pas gérer les éléments spéciaux -> les noeuds sont tracés 2 fois (pas grave !)
"""
def fonct_Affichage(pbEF, mot_clef, colormap = 'coolwarm', vp = 1):
    """
*Affichages :* maillage initial et champ 'mot_clef' sur le maillage déformé (linéaire) et les efforts imposés
** Mot-clefs optionnels **
-* colormap :* couleur de l'affichage de la colorbar affichée pour champ <> 'u''; si vide ne pas afficher 
-* vp :* rang de la valeur propre et du vecteur propres à tracer, si champ = 'X'; rang = 1 par défaut

On raisonne par élément : affichage à partir des informations des éléments tous parcourus;  
pour ne pas gérer les éléments spéciaux -> les noeuds sont tracés 2 fois (pas grave !)

*Mot-clef :*
- 'u': déplacement
- 'X' : mode propre
- 'e': déformation
- 'K' : courbure
- 'N' : effort normal
- 'M' : moment de flexion
- 'ES' : rigidité de membrane
- 'EI' : rigidité de flexion
- Smasse' : masse de la section     

    *Parameters*
    
    mot_clef : char
        Définit le champ à afficher
        
    colormap : 
        définit le style de couleur de la colormap et indique s'il faut l'afficher; chaine vide <=> ne pas afficher
        
    vp : 
        rang de la valeur & vecteur propre à afficher si mot_clef champ = 'X'; 1ère VP par défaut
    *Returns*
   
    ax : figure
        
    """


    F_tot = pbEF.F_tot
    nodes_global = pbEF.nodes_global
    elem_global = pbEF.elem_global
        
    affiche = {'u': [100, 'Ampl. Déplacement [m]'],
               'X': [100, 'Mode Adim embrane [-]'],
               'e': [0, 'Déformation normales [-]'],               
               'K': [1, 'Courbures [m^-1]'],
               'N': [2, 'Efforts Normaux [N]'],
               'M': [3, 'Moments [N.m]'],
               'ES': [4, 'Rigidité <ES> [N]'],
               'EI': [5, 'Rigidité <EI> [N.m^2]'],
               'Smasse': [6, 'Masse linéique [kg.m^(-1)]']               
               }
    RHST = F_tot
    
    nb_nodes = len(nodes_global)
    nb_elem = len(elem_global)
    total_dofs = nb_nodes*3
    plt.close()
    x1_node = []
    x2_node = []
    x1_def = []
    x2_def = []
    num_node = []
    dx1 = np.zeros(nb_nodes)
    dx2 = np.zeros(nb_nodes)
    
    num_elem = []
    x1_elem = []
    x2_elem = []
   
    # Palette de couleurs
    if colormap : 
        cmap = plt.cm.get_cmap(colormap)
    else  :
        cmap = plt.cm.get_cmap('coolwarm')
    fontsize = 4
    markersize = 2
    lw = 0.3
        
    # mise à jour coordonnées des noeuds - utilisée aussi pour déterminer le rapport d'aspect de la figure
    if mot_clef == 'X': # pour les modes propres, mise à jour des déplacements explicitement
        mode_visu = vp - 1
        for nodei in range(len(pbEF.nodes_global)):
            pbEF.nodes_global[nodei].u1 = pbEF.mode_min[nodei * 3, mode_visu]
            pbEF.nodes_global[nodei].u2 = pbEF.mode_min[nodei * 3 + 1, mode_visu]
            pbEF.nodes_global[nodei].theta = pbEF.mode_min[nodei * 3 + 2, mode_visu]   

    for i in range(nb_nodes):
        x1_node.append(nodes_global[i].x1)
        x2_node.append(nodes_global[i].x2)
        
        dx1[i] = nodes_global[i].u1
        dx2[i] = nodes_global[i].u2
    # amplification des déplacements tel que le déplacement maxi représente 5% de la dimension maximal.
    # arrondi à la dizaine. 
    ampli = 0.05* max(max(x1_node),max(x2_node)) / max(np.max(np.abs(dx1)),np.max(np.abs(dx2)))
    if ampli < 10:
        amplification_def = max(1., round(ampli,0))
    else:     
        amplification_def = round(ampli, -1 )      
    
    for i in range(nb_nodes):       
        x1_def.append(x1_node[i] + amplification_def*nodes_global[i].u1) 
        x2_def.append(x2_node[i] + amplification_def*nodes_global[i].u2)
        num_node.append(nodes_global[i].numero)

    # en fonction des champs à afficher
    ind_vecteur_affiche = affiche[mot_clef][0] 
    legende = affiche[mot_clef][1]
        
    
    if ind_vecteur_affiche < 100: # Si affichage grandeurs autres que déformée, on récupère les infos des éléments
        vecteur_affiche = np.zeros(nb_elem)       
        for indice, elem in enumerate(elem_global):
            elem_temp = elem_global[indice]  
            num_elem.append(elem_temp.num_elem)
            x1_elem.append( (1/2) * (elem_temp.node1.x1 + elem_temp.node1.u1*amplification_def + \
                                 elem_temp.node2.x1 + elem_temp.node2.u1*amplification_def))
            x2_elem.append( (1/2) * (elem_temp.node1.x2 + elem_temp.node2.u2*amplification_def + \
                                 elem_temp.node2.x2 + elem_temp.node2.u2*amplification_def))
            if ind_vecteur_affiche < 4 :
                vecteur_affiche[indice] = elem_temp.def_eff[ind_vecteur_affiche]              
            elif ind_vecteur_affiche == 4:
                vecteur_affiche[indice] = elem_temp.ES
            elif ind_vecteur_affiche == 5:
                vecteur_affiche[indice] = elem_temp.EI
            else :
                vecteur_affiche[indice] = elem_temp.Smasse  
            
    else : # déplacements ou modes
        vecteur_affiche = np.zeros(nb_nodes)
        vecteur_affiche = np.sqrt(np.power(dx1,2) + np.power(dx2,2))
    
    # Dimensions de la figure - déplacements min et max, mais aussi taille des flèches
    espace_figure = 1.15
    min_def_x1, min_def_x2, max_def_x1, max_def_x2 = min(x1_def), min(x2_def), max(x1_def), max(x2_def)
    dim_x1, dim_x2 = (max(max(x1_node),max_def_x1) - min(min(x1_node),min_def_x1)), \
        max(max(x2_node),max_def_x2) - min(min(x2_node),min_def_x2)
            
    fig, ax = plt.subplots(figsize=(dim_x1, max(dim_x2/dim_x1,1/3)*dim_x1 ), dpi=400)   
    ax.axis('off')
    ax.set_yticks([])
    ax.set_xticks([])
    
    # dimensions du graphique en coordonnées réelles (physques)
    lim_xmin, lim_xmax, lim_ymin, lim_ymax = min(min_def_x1,-0.2)*espace_figure, max(max(x1_node),max_def_x1)*espace_figure, \
        min(min_def_x2,-0.2)*espace_figure, max(max(x2_node),max_def_x2)*espace_figure
    ax.set(xlim=(lim_xmin,lim_xmax),ylim=(lim_ymin, lim_ymax) )     
    
    # Affichage des numéros d'éléments au centre de l'élément du maillage déformé pour les champs autres
    # que la déformée : pour la déformé num_elem est vide
    for xi, yi, text in zip(x1_elem, x2_elem, num_elem):
        ax.annotate(text, xy=(xi, yi), xycoords='data', xytext=(-4.5, 4.5), textcoords='offset points', 
                    color='black', fontsize=fontsize)
        
    # Maillage initial 
    ax.vlines(0., lim_ymin, lim_ymax,  colors='black', ls='-.',lw=1.5*lw)
    ax.annotate('^ x_2', xy=(-0.05,lim_ymax), xytext=(0, 0), textcoords='offset points',color='black', fontsize=fontsize)
    ax.hlines(0., 0.03, 0.9, transform=ax.get_yaxis_transform(), colors='black', ls='-.', lw=1.5*lw)       
    ax.annotate('> x_1', xy=(max_def_x1,0.), transform=ax.get_xaxis_transform(),  xytext=(4.5, -1.), textcoords='offset points',color='black', fontsize=fontsize)
    ax.annotate('(0,0)', xy=(0.,0.), transform=ax.get_xaxis_transform(),  xytext=(1.5, 1.5), textcoords='offset points',color='black', fontsize=fontsize)
    ax.plot(x1_node,x2_node,'x',markersize = markersize,color='#1f77b4')
    for ind, elem in enumerate(elem_global):                
        plt.plot([elem.node1.x1, elem.node2.x1], [elem.node1.x2 ,elem.node2.x2], color='#1f77b4',ls='--', lw=lw)  
    
    # Affiche les efforts à partir de RHST
    # Trace des fleches pour représenter les efforts et une X pour les moments
    # On connait les DDLs, on en déduit à quel noeuds ils correspondent et leur type
    decal = 1.E-1
    if np.linalg.norm(RHST) !=0 and not pbEF.vibration :
        # La taille des fleches est calculée telle que la plus grande positionnée sur son noeud ne sorte pas de la figure
        Fmax = 0.
        abs_RHST = np.abs(RHST)
        ddl = 0
        node_Fmax = 0
        for ind in range(nb_nodes):
            Fmax1_temp = abs_RHST[3*ind]
            Fmax2_temp = abs_RHST[3*ind+1]
            if (Fmax1_temp > Fmax) : 
                Fmax = Fmax1_temp
                ddl = 1
                node_Fmax = ind
            if (Fmax2_temp > Fmax) : 
                Fmax = Fmax2_temp
                ddl = 2
                node_Fmax = ind

        if Fmax == 0:
            Fmax = 1.
            
        if ddl == 1:
            if (Fmax > 0) :
                scale_force = (lim_xmax-x1_def[node_Fmax])/Fmax
            else :
                scale_force = (-lim_xmin-x1_def[node_Fmax])/Fmax 
        else :
            if (Fmax > 0) :
                scale_force = (lim_ymax-x2_def[node_Fmax])/Fmax
            else :
                scale_force = (-lim_ymin-x2_def[node_Fmax])/Fmax 

        seuil_affichage = 1E-8
        width = 1E-4
        head_width = 0.03
        head_length= 0.08 
        # if mot_clef !='X':
        for i in range(total_dofs):
            if np.abs(RHST[i]) > seuil_affichage*np.linalg.norm(RHST):
                df = scale_force * RHST[i]
                indice_node = i // 3 # division entière
                ddl = i % 3 + 1 
                if ddl == 1:
                    if df > 0:
                        ax.arrow(x1_def[indice_node]+decal, x2_def[indice_node], df, 0.0, width = width,lw =lw ,
                             head_width=head_width, head_length=head_length, fc='blue', ec='blue',length_includes_head=True)
                    else :
                        ax.arrow(x1_def[indice_node]-df+decal, x2_def[indice_node], df, 0.0, lw =lw ,
                             head_width=head_width, head_length=head_length, fc='blue', ec='blue',length_includes_head=True)
                elif ddl == 2:
                    if df > 0:
                        ax.arrow(x1_def[indice_node], x2_def[indice_node], 0.0, df, lw =lw ,
                             head_width=head_width, head_length=head_length, fc='blue', ec='blue',\
                             length_includes_head=True)
                    else :
                        ax.arrow(x1_def[indice_node], x2_def[indice_node]-df+decal, 0.0, df,  lw =lw , 
                             head_width=head_width, head_length=head_length, fc='blue', ec='blue', \
                             length_includes_head=True, head_starts_at_zero=True)
                elif ddl == 3:
                    ax.plot(x1_def[indice_node], x2_def[indice_node], 'X', color='b', markersize = markersize*1.5) 
    
    # 1/ : tracé de la déformée, avec les numéros de noeuds et les numéros des éléments 
    if mot_clef == 'u' or mot_clef == 'X' :
        vecteur_affiche = np.sqrt(np.power(dx1,2) + np.power(dx2,2))
        norm = Normalize(vmin=min(vecteur_affiche), vmax=max(vecteur_affiche)) # fonction pour convertir entre 0 et 1 les valeurs entre ces bornes                  
        # Affichage des numéros de noeuds 
        for xi, yi, text in zip(x1_def, x2_def, num_node):
            ax.annotate(text, xy=(xi, yi), xycoords='data', xytext=(-5., -5), textcoords='offset points', color='r', fontsize=fontsize-2)

       # Déformée
        for ind, elem in enumerate(elem_global):
            # ax.annotate(str(elem.num_elem), xy=(elem.node1.x1, elem.node1.x2), xycoords='data', xytext=(-5., -5), 
            #             textcoords='offset points', color='g', fontsize=fontsize-2)
            ax.plot([(elem.node1.x1 + elem.node1.u1*amplification_def), (elem.node2.x1 + elem.node2.u1*amplification_def)], 
                    [(elem.node1.x2 + elem.node1.u2*amplification_def), (elem.node2.x2 + elem.node2.u2*amplification_def)],
                    color=('r'), ls='-',lw=lw*2)
                
        scale = 10 / amplification_def
        sm = ax.quiver(x1_def,x2_def, dx1, dx2, vecteur_affiche, cmap=cmap, 
                        norm=norm, scale=scale, width=5E-3)
    # 2/ : Eléments colorés en fonction de la valeur de valeur_affiche
    else:      
        norm = Normalize(vmin=min(vecteur_affiche), vmax=max(vecteur_affiche)) # fonction pour convertir entre 0 et 1 les valeurs entre ces bornes                

        for ind, elem in enumerate(elem_global):  
            color_elem = cmap(norm(vecteur_affiche[ind])) # Choix de couleur en fonction de la déformation
            # print("[",elem.node1.x1 + elem.node1.u1,",",elem.node1.x2 + elem.node1.u2,"] \n", 
            #       "[",(elem.node2.x1 + elem.node2.u1),",", (elem.node2.x2 + elem.node2.u2),"]")                            
            ax.plot([(elem.node1.x1 + elem.node1.u1*amplification_def), (elem.node2.x1 + elem.node2.u1*amplification_def)], 
                    [(elem.node1.x2 + elem.node1.u2*amplification_def), (elem.node2.x2 + elem.node2.u2*amplification_def)],
                    color=color_elem, lw=lw*8)


    # 3/ Affichage des valeurs extrêmes 
    ind_max = np.argmax(vecteur_affiche)    
    ind_min = np.argmin(vecteur_affiche)
    if mot_clef == 'u' or mot_clef == 'X':
        ind_min_affiche = 'Min ='+f"{vecteur_affiche[ind_min]:.2e} \n (Noeud {nodes_global[ind_min].numero})"
        ind_max_affiche = 'Max ='+f"{vecteur_affiche[ind_max]:.2e} \n (Noeud {nodes_global[ind_max].numero})"
        ax.annotate(ind_min_affiche, xy=(x1_def[ind_min], x2_def[ind_min]), \
            xycoords='data', xytext=(-5., -18), textcoords='offset points', color='b', fontsize=fontsize)
        ax.annotate(ind_max_affiche, xy=(x1_def[ind_max], x2_def[ind_max]), \
            xycoords='data', xytext=(-5., -18), textcoords='offset points', color='r', fontsize=fontsize)
                    
            
    else :
        ind_min_affiche = 'Min ='+f"{vecteur_affiche[ind_min]:.2e} \n (Elem {elem_global[ind_min].num_elem})"
        ind_max_affiche = 'Max ='+f"{vecteur_affiche[ind_max]:.2e} \n (Elem {elem_global[ind_max].num_elem})" 
        ax.annotate(ind_min_affiche, xy=(x1_elem[ind_min], x2_elem[ind_min]), \
        xycoords='data', xytext=(-5., -18), textcoords='offset points', color='b', fontsize=fontsize)
        ax.annotate(ind_max_affiche, xy=(x1_elem[ind_max], x2_elem[ind_max]), \
        xycoords='data', xytext=(-5., -18), textcoords='offset points', color='r', fontsize=fontsize)

    
    # 3/ gestion de la colorbar si autre chose que déplacement à afficher
    if mot_clef != 'u' and mot_clef != 'X' and colormap :
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)     
        sm.set_array([])
        cbar_ax = plt.gca()
        cbar = plt.colorbar(sm, shrink=0.8, ax=cbar_ax)
        cbar.set_label(legende, fontsize=fontsize)
        cbar.ax.tick_params(labelsize=fontsize)  # Taille des nombres
        cbar.outline.set_linewidth(lw)
        
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-2, 2))
        formatter = ticker.FormatStrFormatter('%.2e')  # Formate les ticks avec 2 décimales en notation scientifique
        cbar.ax.yaxis.set_major_formatter(formatter)
        cbar.update_ticks()  # Mettre à jour les ticks
        # cbar.ax.yaxis.set_major_formatter(formatter)
        # cbar.ax.yaxis.offsetText.set_fontsize(fontsize)
        # cbar.update_ticks()
        # Position de la colorbar
        pos = cbar.ax.get_position()
        cbar.ax.set_position([pos.x0 - 0.05, pos.y0 - 0.0, pos.width * 0.6, pos.height * 0.8])  # Définir la position et la taille de la colorbar
  
        # affichage amplification déformation 
        if amplification_def !=0: 
            ax.annotate('Disp.Mag. ='+str(amplification_def), xy=(pos.x0 - 0.2,pos.y0 - 0.1), xycoords='figure fraction', color='blue', size=fontsize)       

    elif  mot_clef == 'u' and amplification_def !=0: 
        ax.annotate('Disp.Mag. ='+str(amplification_def), xy=(0.6,0.2), xycoords='figure fraction', color='blue', size=fontsize)       
    
    elif mot_clef == 'X':
        if pbEF.vibration : 
            text_leg = 'Pulsation Pr. '
        else :
            text_leg = 'Charge Crit. '
        ax.annotate( text_leg +str(vp) + " = " + f"{pbEF.vp_min[mode_visu]:.2e}", xy=(0.3,0.15), 
                        bbox=dict(
                            boxstyle="round",  # Style de la boîte ("round" pour des coins arrondis)
                            edgecolor="red", # Couleur des bords de la boîte
                            facecolor="lightblue", # Couleur de fond de la boîte
                            linewidth=0.6        # Épaisseur des bords
                        ), xycoords='figure fraction', color='black', size=fontsize)       
             
    return ax


"""
=================================================================================

                        PARTIE 3 : classe pbEF_poutre

=================================================================================

============================== 1/ Maillage, 2/ Def_CL, 3/ Application_CL, 4/ Resol  =================

 Maillage et initialisation (yc création) des grandeurs du problème : rigidités et efforts extérieurs
 
 
Tout est exprimé dans le repère global - les matrices de rigidités et le vecteur des efforts réparti 
sont exprimés littéralement (pas d'intégration numérique) pour tout angle formé entre l'élément et l'axe des x1 (horizontal)
"""
  
"""
======================================  Classe pbEF ===============================
"""
class pbEF_poutre:
    """
**Objet PROBLEME = résolution par EF et post-traitement**

*Méthodes :*
    1. initialisation (maillage et assemblage), 
    2. et 2'/ définition des CL Neumann et Dirichlet prise en compte des conditions de Dirichlet, 
    3. résolution,
    4. calculs déformations et efforts internes : utilise fonction 'Calcul_Def_Eff' définie dans 'BWB_Classes.py'
    5. affichage = utilise fonction 'Affichage' définie dans 'BWB_Classes.py'
    6. mise à jour raideur suite à mise à jour des propriétés physiques d'un élément
    
*Remarque :* la discrétisation spatiale maillage et les conditions aux limites ne changent pas pour une instance pbEF; le maillage EF peut changer !
     
"""

    def __init__(self, nodes_global, nodes_elem, props_elem):   
        self.nodes_global = nodes_global
        self.nodes_elem = nodes_elem
        self.elem_global = []
        self.props_elem = props_elem
        nb_nodes = self.nb_nodes = len(nodes_global)
        
        total_dofs = self.total_dofs = 3*nb_nodes
        self.K_global = np.zeros((total_dofs,total_dofs))
        self.Kgeom_global = np.zeros((total_dofs,total_dofs))
        self.M_global = np.zeros((total_dofs,total_dofs)) 
        self.F_con = np.zeros(total_dofs)
        self.F_rep = np.zeros(total_dofs)
        self.F_tot = np.zeros(total_dofs)
        
        self.fixed_dofs = []
        self.vp_min = []
        self.mode_min = []
        self.vibration = False
 
    def Maillage_Assemblage(self):
        """
       Génére les grandeurs globales (vecteurs, matrices) à partir d'une liste de paire de noeuds (et numéro d'élément associé)

       * Retourne :* les vecteurs et matrices globaux assemblés
       -------------

       """
        fonct_Maillage_Assemblage(self)
           
        return 
                        
    
    def Application_CL(self, forces, deplacements):
        """ Prise en compte des conditions de Neumann et Dirichlet homogènes  
        - vérifie au préalable que les noeuds où sont appliquées les CL existent bien
        - Neumann = vecteurs forces
        - Dirichlet = contributions nulles et pour la diagonale : rigidité unitaire, masse nulle pour ne pas polluer le spectre des VP"""

        fonct_Application_CL(self, forces, deplacements)


    def Resol(self):
        """ Résout le système d'équations linéaires avec les conditions de Dirichlet homogènes imposées 
        
        ..  attribute : K_global, Kgeom_global, F_tot, dofs

            """
         
        dofs = np.zeros(self.total_dofs)
        dofs = np.linalg.solve(self.K_global, self.F_tot)
              
        # met à jour les dofs (u1,u2,theta) dans la structure nodes_global - et donc modifie les DOFS des noeuds attachés aux éléments
        for indice in range(self.nb_nodes):
            self.nodes_global[indice].u1 = dofs[3*indice]
            self.nodes_global[indice].u2 = dofs[3*indice+1]
            self.nodes_global[indice].theta = dofs[3*indice+2]

        return self.nodes_global  
    
    def Calcul_Def_Eff(self):
        """ Calcul des déformations et les efforts généralisés par élément - appel fonct_Calcul_Def_Eff()"""
        fonct_Calcul_Def_Eff(self)
        return self.elem_global   
    
    def MAJ_Rigidite_Geom(self):
        """ Mise à jour des rigidités géométriques connaissant les efforts normaux du problème initilal - appel MAJ_Rigidite_Geom_elem """
        for ind_elem, elem in enumerate(self.elem_global):
            fonct_MAJ_rigidite_geom_elem(self, ind_elem+1)
        return self.elem_global       
    
    def Calcul_Modes_Propres(self, mot_clef, n_modes):
        """ Calcul les n_modes premières valeurs propres et modes associés - appel fonct_Calcul_Modes_Propres(mot_clef, n_modes)"""
        fonct_Calcul_Modes_Propres(self, mot_clef, n_modes)
        return self.vp_min, self.mode_min
    
    def Affichage(self, mot_clef, colormap, vp):
        """ Affichage des résultats - appel fonct_Affichage(mot_clef, colorbar, rang valeur propre) """
        return fonct_Affichage(self, mot_clef, colormap, vp)
    



# def fonct_Assemble_et_CL_Poutre(F_con, px, py, fixed_dofs, pbEF):
#     """
# 1/ Calcul les rigidités et efforts répartis, 
# 2/ Applique les CL Neuman (concentrees) et Dirichlet 
# *Retourne* les structures globales modifiées : rigidités, efforts, liste éléments, ...
#     """
#     # global nodes_global, K_global, Kgeom_global, M_global, elem_global, F_rep, F_tot    

#     nodes_global = pbEF.nodes_global
#     K_global = pbEF.Kglobal
#     Kgeom_global = pbEF.Kgeom_global
#     M_global = pbEF.M_global
#     elem_global = pbEF.elem_global
#     F_rep = pbEF.F_rep
#     F_tot = pbEF.F_con

#     Nl = len(nodes_global)-1

#     for eli in range(Nl):
#         Creation_Elem_Assemblage(eli+1, eli+1, eli+2, Meca_elem, px, py, pbEF)  
 
#     F_tot = F_con + F_rep
                  
#     for dof in fixed_dofs:
#          K_global[dof, :] = 0
#          K_global[:, dof] = 0
#          K_global[dof, dof] = 1

#          Kgeom_global[dof, :] = 0
#          Kgeom_global[:, dof] = 0
#          # proposition de https://bleyerj.github.io/comet-fenicsx/tours/eigenvalue_problems/buckling_3d_solid/buckling_3d_solid.html
#          # si on met 1 pour les CL Dirichlet -> vp =1, polluent le spectre; si on met 0 -> infty
#          Kgeom_global[dof, dof] = 0
         
#          F_tot[dof] = 0        
                          
#     return K_global, Kgeom_global, M_global, elem_global, F_tot




