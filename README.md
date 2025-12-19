# 🐍🧠 SNAKE IA - PACK COMPLET

Trois programmes incroyables pour voir une IA apprendre à jouer à Snake !

## 📦 Les 3 fichiers

### 1️⃣ `snake_ai_complete.py` - LE JEU COMPLET
**Ce que tu vois:**
- 🐍 Le serpent qui joue en direct
- 🍎 Les pommes qu'il mange
- 📊 Génération actuelle
- 🏆 Score, record, moyenne
- 📈 Graphique de progression

**Lancer:**
```bash
python snake_ai_complete.py
```

**Parfait pour:** Voir l'IA s'améliorer partie après partie

---

### 2️⃣ `neural_network_visualizer.py` - LE CERVEAU SEUL
**Ce que tu vois:**
- 🧠 Le réseau de neurones en 3D visuel
- ⚡ Les neurones qui s'allument
- 🌊 Le flux d'information (particules)
- 🎨 Les connexions entre neurones
- 🎭 Différents scénarios de jeu

**Lancer:**
```bash
python neural_network_visualizer.py
```

**Contrôles:**
- `ESPACE` : Changer de scénario
- `A` : Auto-démo ON/OFF
- `Q` : Quitter

**Parfait pour:** Comprendre comment l'IA "pense"

---

### 3️⃣ `snake_ai_with_brain.py` - LES DEUX COMBINÉS ! 🔥
**Ce que tu vois:**
- 🐍 Le serpent qui joue (à gauche)
- 🧠 Son cerveau en action (à droite)
- 📊 Toutes les stats (en bas)
- 🎯 Les décisions en temps réel

**Lancer:**
```bash
python snake_ai_with_brain.py
```

**Parfait pour:** Voir EXACTEMENT comment l'IA décide de ses mouvements !

---

## 🚀 Installation ultra-rapide

```bash
# Installer tout d'un coup
pip install pygame numpy torch

# Puis lancer celui que tu veux !
python snake_ai_complete.py           # Jeu seul
python neural_network_visualizer.py    # Cerveau seul
python snake_ai_with_brain.py          # Les deux ensemble
```

---

## 🎯 Lequel choisir ?

### Tu veux juste voir l'IA jouer ?
→ **`snake_ai_complete.py`**
✅ Simple, clair, avec stats

### Tu veux comprendre le réseau de neurones ?
→ **`neural_network_visualizer.py`**
✅ Visualisation magnifique du cerveau
✅ Différents scénarios à tester

### Tu veux TOUT voir en même temps ?
→ **`snake_ai_with_brain.py`** 🏆
✅ Le jeu + le cerveau synchronisés
✅ L'expérience complète !

---

## 🧠 Comment ça marche ?

### Le réseau de neurones

```
INPUT (11 neurones)          SORTIE (3 actions)
├─ Danger devant?           ├─ Tout droit
├─ Danger droite?           ├─ Tourner droite
├─ Danger gauche?           └─ Tourner gauche
├─ Direction actuelle (4)
└─ Position nourriture (4)
         ↓
    HIDDEN LAYERS
    (256 neurones x2)
    = Le "cerveau"
```

### L'apprentissage

1. **Au début (0-50 parties)**
   - L'IA fait n'importe quoi
   - Score: 0-3
   - Elle explore

2. **Apprentissage (50-200)**
   - Elle comprend les règles
   - Score: 5-15
   - Elle évite les murs

3. **Amélioration (200-500)**
   - Stratégies avancées
   - Score: 15-30
   - Elle évite son corps

4. **Maîtrise (500+)**
   - Performance d'expert
   - Score: 30-70+
   - Survie longue durée

### Les récompenses

- **+10** : Manger une pomme 🍎
- **-10** : Se cogner et mourir 💀
- L'IA apprend de 100 000 expériences !

---

## 🎨 Détails visuels

### Dans `neural_network_visualizer.py`:

- **Neurones verts** = Activation positive (signal fort)
- **Neurones rouges** = Activation négative
- **Particules jaunes** = Information qui circule
- **Glow effect** = Neurone très actif
- **Connexions** = Plus épaisses si actives

### Dans `snake_ai_with_brain.py`:

- Le serpent **à gauche** joue normalement
- Le cerveau **à droite** montre ses décisions
- L'action choisie est **en vert**
- Tu vois la correspondance en **temps réel** !

---

## 💡 Conseils

### Pour un entraînement rapide:
1. Lance `snake_ai_complete.py`
2. Laisse tourner 500+ générations
3. Regarde le record exploser !

### Pour comprendre l'IA:
1. Lance `neural_network_visualizer.py`
2. Appuie sur ESPACE pour voir différents cas
3. Observe quels neurones s'allument

### Pour l'expérience ultime:
1. Lance `snake_ai_with_brain.py`
2. Regarde le serpent ET son cerveau
3. Comprends chaque décision !

---

## 🎮 Contrôles communs

Tous les programmes:
- **ESPACE** : Pause
- **Q** : Quitter

`neural_network_visualizer.py` en plus:
- **A** : Auto-démo

---

## 📊 Progression typique

```
Génération    Score moyen    Ce qu'elle fait
─────────────────────────────────────────────
0-50          0-3            Explore, se cogne
50-100        3-8            Évite les murs
100-200       8-15           Cherche la nourriture
200-500       15-25          Stratégies avancées
500-1000      25-40          Presque parfaite
1000+         40-60+         Experte !
```

---

## 🎉 Profite du spectacle !

Tu as maintenant 3 façons différentes de voir une IA apprendre toute seule !

**Mon préféré ?** `snake_ai_with_brain.py` - c'est magique de voir le cerveau et le jeu ensemble ! 🤯

---

## 🐛 Problèmes ?

**"pygame not found"**
```bash
pip install pygame
```

**"torch not found"**
```bash
pip install torch
```

**"Trop lent"**
→ Augmente `FPS` dans le code (ligne ~15)

**"Trop rapide"**
→ Diminue `FPS` dans le code

---

## 🚀 Et après ?

Une fois que ton IA est entraînée:
- Elle peut atteindre des scores de 50-70+
- Regarde-la devenir de plus en plus intelligente
- Compare les différents entraînements
- Modifie les paramètres et réentraîne !

**Amusez-vous bien ! 🎮🧠**
