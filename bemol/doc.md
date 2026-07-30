# DOCUMENTATION SUCCINTE DE BEMOL
## Classe rotor

Classe permettant de gérer les éoliennes et de manipuler leur géométrie. Un objet rotor contient : 

    - Le rayon associé à chaque section de la pale
    - Le twist associé à chaque section
    - Le chord associé à chaque section
    - Les interpolées des fonction de lift et de drag

Pour construire un objet rotor (utile pour debugger si jamais), il faut spécifier le chemin vers un dossier contenant les paramètres du rotor (un rotor.yml et un blade.dat ainsi qu'un sous dossier contenant les profils).

## Classe BASEbem
C'est la classe fondamentale du code. C'est dans celle que sont stockées
 
    - Un objet rotor
    - La densité de l'air
    - Une liste de modèle correctifs implémentée pour bemol (voir bemol/bemol/secondary)

## Classe NingUncoupled
Classe dérivée de BASEbem (donc elle peut s'initialiser avec la donnée des attributs de la classe BASEbem). C'est sur cet objet que l'on va effectuer tous les calculs de résolution de la BEM. Elle a pour attributs  : 

    - La vitesse 2 [U_x, U_y]
    - L'induction axiale 
    - L'induction tangentielle
    - Les interpolées de C_L et C_D
    - Un attribut epsilon servant à) la résolution par la méthode de Ning (pas la peine de le toucher)
    - Tous les attributs de la classe BASEbem (donc en particulier un objet rotor)

Il y a plusieurs maniières de résoudre la BEM avec la classe NingUncoupled. 

### 1. Résolution sur plusieurs révolutions 
La méthode en question est cycle de la classe BASEbem. Elle résout la BEM sur un nombre de révolution de la pale (par défaut 1) sur une discrétisation azimutale spécifiée par le paramètre de discrétisation n_phi.

### 2. Résolution directe
Pour résoudre directement la BEM, c'est-à-dire pour un angle azimutale et une section donnée, c'est la méthode solve de NingUncoupled qui rentre en jeu. Pour la donnée de la vitesse, d'un objet solver, de quelques paramètres mécaniques (comme le tilt ou le precone), d'un yaw, d'une liste de section et d'un angle azuimutale elle retourne l'induction et l'effort axial ainsi que l'induction et l'effort tangentiel. C'est dans cette fonction que les modèles correctifs de BASEbem sont appelés.

### 3. Résolution pour une configuration météorologique donnée
La méthode steady de BASEbem est similaire à la fonction solve de NingUncoupled. Celle-ci précalcule tous les attributs de la classe NingUncoupled (comme la vitesse)  avant d'appeler solve.

## Classe SimEnv

La classe SimEnv permet d'intialiser des objets contenant tous les paramètres de la simulation, à savoir : 

    Les paramètres mécaniques : 
        - Le precone
        - Le tilt
        - Le pitch
        - Le rotor ainsi que le chemin menant au rotor
        - La vitesse de rotation de l'éolienne

    Les paramètres météorologique : 
        - La vitesse freestream
        - La densité de l'air
        - le yaw ainsi qu'un booléen act_yaw 
        - l'angle de sillage (skew)
    
    Des paramètres de simulations : 
        - Le nombre de révolution
        - Le TSR
        - Le pas de temps pour la BEM non stable

La classe a également un attribut solver (de la classe NingUncoupled) et une liste contenant tous les modèles de corrections que l'on souhaite charger. 

Les fonctions d'intérêts de cette classe sont data_maker et para_data_maker. Ces deux méthode font la même choses : pour une liste de TSR et de Yaw donnée, elles calculent les champs de forces BEM associés à un couple (TSR, yaw). La différence est que data_maker est une méthode séquentielle, tandis que para_data_maker est parallèle. Elles retournent toutes les deux un dataframe conçu de la même manière que celui utilisée pour stocké les données SVEN, avec une colonne angle d'attaque, vitesse effective, force normale et force tangentielle.