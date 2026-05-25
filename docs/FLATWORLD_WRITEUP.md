# FlatWorld Environments: Teacher-to-Student Transfer

## Overview

The FlatWorld environment is a two-dimensional continuous world (𝒮 = [−2, 2]²) with a discrete action space and colored circular regions representing atomic propositions. Importantly, these regions overlap in various places, which means that multiple propositions can hold true at the same time. The initial agent position is sampled randomly from the space in which no propositions are true. At each time step, the agent can move in one of the 8 compass directions. If it leaves the boundary of the world, the agent receives a penalty and the episode is terminated prematurely.

## FlatWorld Patrol: Two-Color Navigation with Obstacles

### Task Definition

The patrol task requires the agent to visit two target locations sequentially, satisfying the temporal logic formula: **F(b ∧ F(a))** ("eventually reach region *b*, then eventually reach region *a*").

**Proposition Mapping** (alphabetically sorted):
- **a** ↔ <span style="color: #1e88e5">**BLUE**</span> region
- **b** ↔ <span style="color: #e53935">**RED**</span> region

**Execution Order**: Visit red (b) → then blue (a)

### Teacher Configuration (Source Task)

- **Circles**: Default layout with 9 overlapping colored regions distributed spatially
- **Walls**: None (open environment)
- **Difficulty**: Low — clear visibility and direct path planning

The teacher learns to navigate this simplified environment without obstacles, establishing a baseline strategy for reaching both target regions.

### Student Configuration (Target Task)

- **Circles**: Shifted layout — same propositions but relocated to different spatial positions
- **Walls**: Cross-shaped obstacle (vertical + horizontal bars intersecting at origin)
- **Difficulty**: Medium — shifted goals require re-learning positions + walls force path detours

The student must transfer knowledge of the patrol strategy *despite* goal relocation and new physical obstacles. The cross wall divides the space into four quadrants, forcing the agent to plan around the obstacle.

### Transfer Challenge

The spatial shift of circles tests whether the agent can generalize the *behavioral pattern* (visit region types in sequence) across different spatial configurations. The addition of a cross wall introduces **spatial reasoning** challenges: the agent must discover detour paths around the obstacle while maintaining the sequential visitation strategy learned from the teacher.

---

## FlatWorld Sequence: Three-Color Navigation with Corridors

### Task Definition

The sequence task requires visiting three target regions in order, satisfying: **F(c ∧ F(a ∧ F(b)))** ("eventually reach region *c*, then *a*, then *b*").

**Proposition Mapping** (alphabetically sorted):
- **a** ↔ <span style="color: #1e88e5">**BLUE**</span> region
- **b** ↔ <span style="color: #43a047">**GREEN**</span> region  
- **c** ↔ <span style="color: #e53935">**RED**</span> region

**Execution Order**: Visit red (c) → blue (a) → green (b)

### Teacher Configuration (Source Task)

- **Circles**: Default layout (same as patrol teacher)
- **Walls**: None (open environment)
- **Difficulty**: Low — horizonally extended task but without physical constraints

The teacher learns a three-hop navigation strategy in an unobstructed world, providing knowledge about region visitation order and general navigation.

### Student Configuration (Target Task)

- **Circles**: Shifted layout (same as patrol student)
- **Walls**: Two horizontal corridor walls with opposite-side gaps
  - Upper wall spans left to center with gap on right
  - Lower wall spans center to right with gap on left
  - Forces a zig-zag path through the corridors
- **Difficulty**: Higher — three goals + corridor navigation complexity

The student faces a longer temporal logic formula (three goals vs. two) combined with spatial constraints that enforce a specific zig-zag traversal pattern. This tests whether transfer knowledge can help with *both* sequential planning and complex spatial reasoning.

#### Corridor Structure Visualization

The corridor layout creates a **forced zig-zag navigation pattern**:

```
        TOP (+2.0)
          ▲
          │
     ┌────┴─────────┐  ← Upper wall (gap on right)
     │  REGION a    │
     │              │
     │    ┌─────────┖  ← Gap forces right turn
─────┼────┤
     │    │ REGION b
     │    └─────────┐
     └────┬─────────┘  ← Lower wall (gap on left)
          │
    ◇ ← Agent starts here (before visiting any regions)
          │
      LEFT(-2) ─────► RIGHT(+2)
           BOTTOM (-2.0)
```

**Forced path**: 
1. Start → Move UP through opening
2. Reach region *a* → Turn RIGHT (upper wall blocks direct path)
3. Descend → Move DOWN through right-side corridor
4. Reach region *b* → Turn LEFT (lower wall blocks direct exit)
5. Continue to region *c*

This corridor architecture ensures the agent **cannot take direct paths** and must learn spatial detours, making transfer significantly more challenging than the open-environment teacher task.

### Transfer Challenge

Unlike patrol (which adds one obstacle), sequence involves both:
1. **Temporal complexity**: one additional goal to visit
2. **Spatial complexity**: **corridor walls that force non-linear, zig-zag paths** — the agent cannot proceed straight to each region and must discover detour routes through specific corridor gaps
3. **Position uncertainty**: shifted circles in the new layout

The corridor design creates a *structural bottleneck*: to visit regions in order, the agent must:
- Navigate UP through the upper corridor gap
- Navigate DOWN through the right-side corridor (forced right turn)
- Navigate LEFT to exit the lower corridor (forced left turn)

This forced zig-zag pattern is markedly different from the teacher's open-world navigation, making spatial transfer a critical test of knowledge adaptation.

---

## Spatial Reasoning and Generalization

Both patrol and sequence require agents to:

1. **Learn region identities**: Distinguish between colored propositions in the environment
2. **Memorize visitation order**: Recall the temporal logic requirements (a → b or a → b → c)
3. **Plan paths**: Navigate around obstacles and shifted positions
4. **Generalize**: Apply learned strategies to new spatial layouts

The **difficulty progression** from teacher to student is designed to be within-domain (same environment, different configuration) rather than cross-domain, making it tractable for transfer learning while still presenting significant challenges.

---

## Visualizations

See Figure 1 (Patrol) and Figure 2 (Sequence) for side-by-side spatial layouts of teacher and student environments. The visualizations show:
- **Red circles**: Proposition *c* (in Sequence) / Proposition *b* (in Patrol)
- **Blue circles**: Proposition *a* (in both tasks)
- **Green circles**: Proposition *b* (in Sequence only)
- **Dark rectangular walls**: Obstacles present only in student tasks
- **World boundary**: ±2 in both dimensions

Color-coded target propositions match across both tasks: <span style="color: #1e88e5">blue (a)</span> always marks a critical waypoint, while <span style="color: #e53935">red</span> and <span style="color: #43a047">green</span> mark later sequential targets. The spatial shift between teacher and student circle positions is evident, demonstrating the non-trivial transfer challenge.
