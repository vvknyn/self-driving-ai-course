# The road's own coordinates

You are the referee. The provided car is driving its lap of `gentle` at 8 m/s, and at every step, twenty times a second, someone has to write down two numbers about it. One is how far along the road it has got, which decides when the lap is finished. The other is how far it is from the centre of its lane: the lateral error $e$ that *Meet your car* graded. The simulator has been the referee all along, and every grade in this course comes from its two numbers. This lesson is about how it gets them.

How far along the road is the car, and how far from its lane centre?

By the end you will write `to_road_frame` and `lane_errors`, two short functions that turn a position in the world into the road's own pair of coordinates. They will reproduce the simulator's lateral error and heading at every one of the 1778 steps of a lap, with a difference of exactly zero.

![Two stacked panels of the provided car on gentle at step 400, 20.00 s into its lap, seen from above. Top: the road from its start into the first bend, where a grey line follows the yellow line from the start of the road to the point on it nearest the car, labelled 159.011 m along the road. Bottom: a zoom on a few metres around the car, where a blue brace from the lane centre to the car's front axle is labelled 0.528 m left of the lane centre.](figures/result_first.png)

Those are the referee's two numbers at one step: 159.011 m along the road, 0.528 m left of the lane centre. Neither is a world coordinate. In the world frame of *Positions and headings* the car's front axle is at x = 154.659 m, y = 19.854 m, and neither 159.011 nor 0.528 can be read off those two numbers. The rest of this lesson builds the road's coordinates, and the code that computes them, one step at a time.

## The lateral error you would guess

The world frame's x axis runs along the road's first straight, its y axis to the left, and the car starts at (0, −1.8), on the lane centre and facing east. On that straight the lane centre is the line y = −1.8. So a first guess for the lateral error is

$$\begin{aligned}e &\approx y + 1.8\end{aligned}$$

A car at y = −1.5 is 0.3 m north of the line y = −1.8, which on the first straight is 0.3 m to its left, and the guess says +0.3 m.

**Guess first:** over the whole lap, how far from the true $e$ does this guess get?

![Two stacked panels for the provided car's lap of gentle, both against time in seconds. Top, the whole lap: the guess y + 1.8, in grey, climbs from 0 to 163.6 m at 53.80 s, on the far straight, while the true lateral error e, in blue, stays near 0 all the way round. Bottom, a zoom on 10 s to 15 s with a scale of a few metres: the two lines lie on top of each other until 12.50 s, when the car reaches the first bend, and then part, 0.01 m apart at 12.70 s, 0.1 m at 13.05 s and 1 m at 14.10 s.](figures/naive_y.png)

**Answer:** 163.6 m, on the far straight, 53.80 s in. On the first straight the guess is exact: it differs from the true $e$ by 0.0 m at every step up to 12.50 s. It goes wrong as soon as the road bends: 0.01 m out at 12.70 s, only 1.6 m into the bend, and 1 m out by 14.10 s. On the far straight it is absurd. At 45.00 s the car is at y = 161.55 m, the guess says it is 163.35 m left of its lane centre, and the true $e$ is 0.248 m.

The guess measures sideways along the world's y axis. That direction is the car's left only while the road points east. Once the road turns, its left turns with it. On the far straight the road runs back west, and the left of a car facing west is south, the direction of −y.

A cleverer guess fixes the lane centre's position. Instead of −1.8, subtract the y of the point of the lane centre nearest the car. It is still a difference of world y values, and it still fails.

**Guess first:** near the end of the first bend, where the road runs almost due west, will the cleverer guess be too big, too small, or the wrong sign?

![The cleverer guess, the car's y minus the y of the nearest point of the lane centre, in grey, and the true lateral error e, in blue, over the lap of gentle against time. On the first straight the two lines agree. The largest gap is at 43.30 s, near the end of the first bend, where the guess is −0.528 m and e is +0.531 m.](figures/naive_nearest_y.png)

**Answer:** the wrong sign. At 43.30 s the guess is −0.528 m while the car is +0.531 m left of its lane centre, 1.059 m apart, almost exactly twice $e$. With the road running west, "more y" means further to the car's right, so the difference in y comes out with the sign flipped.

Both guesses fail for one reason: sideways is not a direction of the world. It has to be measured from the road itself, from the road's nearest point, at right angles to the road's direction there. To do that, you need the road written down.

In the simulator a road is a list of points along the middle of its yellow paint, the **centreline**. It is stored as an (N, 2) array named `centerline`, one row (x, y) per point, the layout of *The view from the driver's seat*. Joining each point to the next with a straight line gives a chain of short straight pieces, called a **polyline**. Each point is a **vertex**, and each straight piece between two neighbouring vertices is a **segment**.

![Two stacked panels. Top: the gentle loop seen from above, drawn as its centreline, with a small box, not to scale, marking 1 m of the first bend. Bottom: a zoom on that box, where the centreline is 11 grey dots, the vertices, 0.1 m apart and joined by short straight segments.](figures/polyline_zoom.png)

`gentle`'s centreline has 7028 vertices, between 0.099971 m and 0.1 m apart. It is **closed**: its last vertex joins back to its first, so it also has 7028 segments, and their lengths add up to 702.65 m. `curvy` is **open**: it has two ends, 2591 vertices and 2590 segments. The distance from the car to the road is its distance to the nearest point of one of these segments. So start with one segment.

## Projecting onto a segment

Take one segment, from vertex $\mathbf{a}$ to vertex $\mathbf{b}$, and a point $\mathbf{p}$. Which point of the segment is nearest $\mathbf{p}$, and how far away is it?

Work it with hand numbers: $\mathbf{a} = (2, 1)$, $\mathbf{b} = (10, 7)$ and $\mathbf{p} = (5, 7)$. The segment's arrow is $\mathbf{b} - \mathbf{a} = (8, 6)$. Its length, by Pythagoras, is $\sqrt{8^2 + 6^2} = 10$. Divide the arrow by its length and you get an arrow 1 m long along the segment, $\hat{u} = (0.8, 0.6)$, read "u-hat". It is a unit vector, like the heading arrow of *Positions and headings*: it is the heading arrow of someone standing at $\mathbf{a}$ and facing $\mathbf{b}$.

Now ask, as *Positions and headings* did for a cone, how far ahead of that someone $\mathbf{p}$ is, and how far to the left. The arrow from $\mathbf{a}$ to $\mathbf{p}$ is $\mathbf{p} - \mathbf{a} = (3, 6)$. Its forward part is a dot product with $\hat{u}$, and its left part a 2D cross product:

$$\begin{aligned}\text{forward} &= (\mathbf{p}-\mathbf{a})\cdot\hat{u}\\ &= 3(0.8) + 6(0.6)\\ &= 6.0\\ \text{left} &= \hat{u}\times(\mathbf{p}-\mathbf{a})\\ &= 0.8(6) - 0.6(3)\\ &= +3.0\end{aligned}$$

**Guess first:** where on the segment is the point nearest $\mathbf{p}$, and how far is $\mathbf{p}$ from it?

![The hand example on a grid in metres. A grey segment runs from a at (2, 1) to b at (10, 7), 10 m long. The point p, at (5, 7), sits to its left. A grey brace along the segment, from a to the foot at (6.8, 4.6), is labelled forward part 6.0. A blue line from the foot to p, at right angles to the segment, is labelled left part 3.0.](figures/segment_hand.png)

**Answer:** 6 m along the segment from $\mathbf{a}$, at $\mathbf{a} + 6\,\hat{u} = (6.8, 4.6)$, and 3.0 m away, exactly the left part. Walking 6 m forward from $\mathbf{a}$ brings you level with $\mathbf{p}$, and what is left is a step straight to the left.

That nearest point is called the **foot**, because the line from it to $\mathbf{p}$ stands at a right angle on the segment, like a foot on the ground. Its distance along the segment from $\mathbf{a}$ is written $t$, read "tee". Here $t = 6.0$.

<details>
<summary>How we got: the foot is the nearest point</summary>

Every point of the segment's straight line is $\mathbf{a} + \tau\,\hat{u}$ for some distance $\tau$ (the Greek letter tau, read "tau") along it from $\mathbf{a}$. Seen from that point, $\mathbf{p}$ is $6 - \tau$ further forward and still 3 to the left, because moving along $\hat{u}$ never changes the left part. By Pythagoras its squared distance is

$$\begin{aligned}D^2 &= (6 - \tau)^2 + 3^2\end{aligned}$$

The $3^2$ never changes, and $(6 - \tau)^2$ is smallest, zero, at $\tau = 6$. So the nearest point is the one level with $\mathbf{p}$, and its distance is the left part, 3. With any numbers, 6 is the forward part and 3 is the left part.

</details>

You have met this pair before. It is the driver's-seat view of *The view from the driver's seat*, for a driver standing at $\mathbf{a}$ and facing along the segment. The segment's direction is atan2(6, 8) = 0.6435 rad, and `world_to_ego((2, 1, 0.6435), p)` gives (6, 3): $t$ ahead and 3 m to the left.

The rule has a flaw at the ends. Try a second point, $\mathbf{q} = (14, 5)$:

$$\begin{aligned}\mathbf{q}-\mathbf{a} &= (12,\ 4)\\ \text{forward} &= 12.0\\ \text{left} &= -4.0\end{aligned}$$

The forward part, 12 m, is longer than the segment, 10 m.

**Guess first:** is $\mathbf{q}$ really 4.0 m from the segment?

![Two stacked panels for the point q at (14, 5) and the same segment from a (2, 1) to b (10, 7). Top, without the clamp: the foot lands at (11.6, 8.2), past the end b, on a dashed grey extension of the segment, and the line to it is 4.0 m long. Bottom, with the clamp: the forward part is cut to the segment's length, 10, the foot is b itself, and the blue line from b to q is 4.472 m long, the true distance to the nearest point of the segment.](figures/clamp.png)

**Answer:** no. 4.0 m is the distance to (11.6, 8.2), a point on the segment's line 2 m past $\mathbf{b}$, where there is no road. The nearest point that is on the segment is its end $\mathbf{b}$, and $\mathbf{q}$ is $\sqrt{4^2 + 2^2} = \sqrt{20} = 4.472$ m from it. The rule needs one more step: **clamp** $t$, which means cutting it to 0 if it is negative and to the segment's length if it is longer. In NumPy that is `np.clip(t, 0, length)`. For $\mathbf{q}$, $t = 12$ becomes $t = 10$, and the foot becomes $\mathbf{b}$.

A distance is never negative, so it cannot say which side $\mathbf{q}$ is on. The left part can: it is −4.0, so $\mathbf{q}$ is to the right of the segment's line. Your first function, `project_to_segment(a, b, p)`, returns two numbers: $t$, and the distance from the point to its foot, given the sign of the left part. For $\mathbf{p}$ it returns (6.0, +3.0); for $\mathbf{q}$ it returns (10.0, −4.4721). Finding the foot is called **projecting** the point onto the segment.

## Which segment, and which side

A road has thousands of segments, and each one gives the point its own foot. The road's nearest point is the nearest of all those feet. Try a small road of two segments, an L: from (0, 0) east to (10, 0), then north to (10, 10). Take $\mathbf{p} = (8, 3)$, inside the corner.

**Guess first:** which segment's foot is nearer $\mathbf{p}$?

![An L-shaped road of two grey segments, from (0, 0) east to (10, 0) and then north to (10, 10). The side to the road's left is shaded light grey and labelled left, plus; the side to its right is labelled right, minus. The point p at (8, 3) has two candidate feet: on the first segment at (8, 0), 3.0 m away, drawn dashed; on the second at (10, 3), 2.0 m away, drawn solid blue. A second point, (12, −2), outside the corner, is 2.828 m from the shared vertex (10, 0), which is the foot on both segments.](figures/l_corner.png)

**Answer:** the second. Its foot, (10, 3), is 2.0 m away, against 3.0 m for the first segment's foot at (8, 0). You do not need to know in advance which segment is nearest. Project the point onto every segment, then pick the smallest distance. The position of the smallest number in a list is called its **argmin**, read "arg min", and NumPy's `np.argmin` finds it. The second segment runs north, so its left is west, and $\mathbf{p}$ is west of it: the left part is positive, and the point is 2.0 m to the left.

That signed distance from the nearest point of the centreline is the road's sideways coordinate. It is written $\color{#58C4DD}{d}$, read "dee": metres to the left of the centreline, negative to the right. Careful: in *Your first perception box*, $d_1$ and $d_2$ were distances **forward**, from the car to the two lane-centre points ahead of it. This $d$ is **sideways**. On `gentle`, $d = 0$ is the middle of the yellow line, your lane centre is $d = -1.8$ m, and the white line is $d = -3.6$ m.

![The car at the start of gentle, seen from above, facing east along the first straight on its lane centre. Two coral dots on the lane centre ahead mark the perception box's two lane-centre points, with coral braces along the lane labelled d₁ = 2.64 m and d₂ = 7.19 m: distances forward. A blue brace across the road, from the yellow line to the car, is labelled d = −1.8 m: a distance sideways.](figures/d_sideways.png)

Outside the corner the two segments tie. The point (12, −2) is 2.828 m from their shared vertex, which is the foot on both. Both give the same foot, so both give the same $d$, −2.828 m. The last section comes back to ties.

On `gentle` there are 7028 segments. On the laptop that wrote this lesson, a Python loop that projected one position onto each segment in turn took between a twentieth and a tenth of a second, depending on what else the laptop was doing, so a lap of 1778 positions would take minutes. Written with whole (N, 2) arrays, the way *Images are arrays* did arithmetic on every pixel at once, the same steps on all 7028 segments took under a millisecond per position, a few hundred times faster. Your times will differ; the ratio is the point. The array version is the one the lab asks for.

## How far along: arc length

The second coordinate says how far along the road the point is. Start at the first vertex of the centreline and walk along it, segment by segment, to the foot. The distance walked is the **arc length** to the foot, written $s$, read "ess". It has two parts: the lengths of all the segments before the nearest one, added up, and $t$, how far along the nearest one the foot is.

![The L-shaped road again, with arc length marked along it in grey: s = 0 at (0, 0), s = 10 at the corner (10, 0) and s = 20 at the far end (10, 10). The point p at (8, 3) has its foot 3 m along the second segment, shown by a grey brace labelled t = 3 m, so its s is 10 + 3 = 13 m. A blue line from the foot to p is labelled d = +2.](figures/arc_length_l.png)

The total length of the segments before segment $i$ is written $s_{0,i}$, read "s nought i". `gentle`'s first straight is 1000 segments of exactly 0.1 m, so segment 1000, the first one of the bend, starts at $s_{0} = 100.0000$ m. NumPy's `np.cumsum` builds every running total in one call. Then

$$\begin{aligned}s &= s_{0,i} + t\end{aligned}$$

In the L, the first segment is 10 m long, so $\mathbf{p} = (8, 3)$, 3 m along the second segment, has $s = 10 + 3 = 13$ m, and its road coordinates are $(s, d) = (13, +2)$.

**Guess first:** at step 400, 20.00 s in, the car has driven 8 m/s × 20.00 s = 160 m. Is its $s$ more than 160 m, or less?

![A few metres of gentle's first bend around the car at step 400, 20.00 s in, with the centreline's vertices as grey dots along the yellow line. The nearest segment, number 1590 of 7028, starts at a, 158.983 m along the road. The foot sits 0.0279 m along that segment, so s is 159.011 m. A blue line at right angles from the foot to the car is labelled d = −1.2717 m: the car is 1.27 m to the right of the yellow line.](figures/step400_foot.png)

**Answer:** less. The nearest segment is number 1590, which starts at $s_{0} = 158.983$ m, and the foot is $t = 0.0279$ m along it, so $s = 159.011$ m while the car has driven 160.000 m. $s$ is measured along the yellow line. On a left bend your lane is on the outside of the turn, on a bigger circle: at step 400 the car is 81.2717 m from the bend's centre, (100, 80), while the yellow line is 80 m from it. Over the whole lap the car drives 710.80 m while $s$ runs from 0 to 702.65 m, and the lane centre's own loop, two 100 m straights and a full circle of radius 81.8 m, is $200 + 2\pi \times 81.8 = 713.96$ m long.

`gentle` is a loop, so a foot just past its last vertex is also just past its first. On a closed road $s$ is taken as the remainder after dividing by the road's length, `s % 702.65`, where `%` is Python's remainder: it brings $s$ back to 0 at the start line. An open road like `curvy` needs no remainder, because the clamp at its two ends already keeps $s$ between 0 and its length.

**Guess first:** the world frame's grid lines are straight. Draw the lines of constant $s$ and of constant $d$ over the first bend. What shape are they?

![The road frame drawn over gentle's first bend, from above. Grey rungs cross the road at right angles every 10 m of s, fanning out from the bend's centre at (100, 80). Three lines of constant d run along the road as circles around that centre: d = 0, the yellow line, 80 m from the centre; d = −1.8, the lane centre, dashed grey, 81.8 m out; and d = −3.6, the white line, 83.6 m out.](figures/road_grid.png)

**Answer:** they bend with the road. The rungs of constant $s$ point at the bend's centre, and every line of constant $d$ is a circle around it, 80 m minus $d$ from it: the car at step 400 has $d = -1.2717$, and 80 + 1.2717 = 81.2717 m. In these coordinates your lane is a straight strip, $d$ from −3.6 to 0, all the way round the loop. The pair $(s, d)$ is the **road frame**: coordinates attached to the road, not to the world or to the car.

The simulator's `tel.progress_m` adds up the change in $s$ from each step to the next, and the lap counts as complete once the total reaches the road's length. This lap ends at 702.699 m, just past 702.65 m. Adding up changes across the start line needs care; the challenge problems ask how.

## The two errors the controller needs

The lane centre is at $d = -1.8$, and $e$ is the distance to the left of the lane centre, so $e$ is just $d$ measured from −1.8 instead of from 0:

$$\begin{aligned}\color{#58C4DD}{e} &= d - (-1.8)\\ &= d + 1.8\end{aligned}$$

At step 400, $d = -1.2717$ m, so $e = +0.5283$ m: the car is 0.53 m left of its lane centre, toward the inside of the bend, as *Meet your car* found.

**Guess first:** the car is off the road once $|e| > 1.8$ m. Which values of $d$ is that?

![Two number lines, one above the other, for the same sideways positions. The top one is d, metres from the yellow line; the bottom one is e, metres from the lane centre, shifted by 1.8. The white line sits at d = −3.6, e = −1.8; the lane centre at d = −1.8, e = 0; the yellow line at d = 0, e = +1.8. The car at step 400 is a blue dot at d = −1.2717 and e = +0.5283. Grey hatching marks off-road beyond both lines.](figures/e_from_d.png)

**Answer:** $d > 0$, over the yellow line, or $d < -3.6$, over the white line. Every $d$ maps to its $e$ by the same shift of 1.8 m.

The second error is the heading. *Positions and headings* gave the car's yaw $\theta$, read "theta": where the car points, measured anticlockwise from east. The road has a direction too: the direction of the nearest segment, which is atan2 of its $\hat{u}$. Call it $\theta_{\text{road}}$, read "theta road". θ is where the car points; ψ, the heading, is the yaw minus the road's direction, wrapped:

$$\begin{aligned}\color{#D147BD}{\psi} &= \text{wrap}(\theta - \theta_{\text{road}})\end{aligned}$$

At step 400, $\theta = 0.7040$ rad and $\theta_{\text{road}} = 0.7379$ rad, so $\psi = -0.0339$ rad: the car points 0.034 rad, about 2°, to the right of the road.

Both errors are what a driver standing at the foot, facing along the road, would see. That driver's pose is the foot with the angle $\theta_{\text{road}}$.

![The car at step 400 seen from a frame placed at the foot on the yellow line, with its x axis along the road and its y axis to the road's left, both drawn in grey. The car's front axle sits at forward 0, left −1.2717, labelled (0, d). Its heading arrow is turned 0.0339 rad clockwise from the frame's x axis, labelled ψ = −0.0339 rad, in pink.](figures/foot_frame.png)

From that frame, `world_to_ego` puts the car at ($-6 \times 10^{-15}$, −1.2717). The first number is zero up to rounding, because the foot is level with the car by construction, and the second is $d$. The road frame is the driver's-seat view taken from the road's nearest point, wherever the car is.

Is the wrap needed? $\theta$ and $\theta_{\text{road}}$ are both directions, and the car never points far from the road.

**Guess first:** leave out the wrap and take the plain difference $\theta - \theta_{\text{road}}$. On how much of the lap is it wrong?

![The heading ψ over the lap of gentle against time, computed two ways. In pink, wrap of theta minus theta road stays within 0.057 rad of 0 all lap. In grey, the plain difference theta minus theta road agrees with it until 56.95 s, then jumps up by 6.2832 rad, a whole turn, and stays there to the end of the lap at 88.85 s.](figures/wrap_trap.png)

**Answer:** on 639 of the 1778 steps, 35.9 % of the lap, it is wrong by a whole turn, $2\pi$ = 6.2832 rad. The simulator never wraps the car's yaw: it adds up each step's turn, and it ends the lap at $\theta = 6.226$ rad, a whole turn less 0.057 rad. atan2, as *Positions and headings* showed, always answers between −π and π. At 56.95 s, as the road turns past due west, atan2's answer for the road jumps from near +π to near −π: $\theta_{\text{road}} = -3.1385$ rad while $\theta = 3.1673$ rad. Their plain difference is +6.3057 rad; wrapped, it is +0.0225 rad. From that step to the end of the lap the plain difference is a whole turn too big.

Your second function, `lane_errors(centerline, closed, pose)`, takes the car's pose $(x, y, \theta)$ and returns $(e, \psi)$. It finds the nearest segment the way `to_road_frame` does, then uses that one segment for both: its foot gives $d$, and its direction gives $\theta_{\text{road}}$.

**Guess first:** run both functions on every step of the provided car's lap. How close do they come to the simulator's own `tel.lat` and `tel.head_err`, its records of the true $e$ and $\psi$?

![Two stacked panels over the lap of gentle against time. Top: the lateral error from lane_errors, in blue, drawn over the simulator's tel.lat, in grey; the blue line lies exactly on the grey one, which shows only as an edge around it, peaking at +0.533 m in the bends. Bottom: the differences between the two functions' four numbers and the simulator's, s, d, e and ψ, all a flat line at exactly 0.](figures/reproduced.png)

**Answer:** exactly. On all 1778 steps the largest difference is 0.0, for $s$, $d$, $e$ and $\psi$ alike. This is not an approximation of the referee: it is the referee's own measurement. The simulator does the same arithmetic in the same order, so the same numbers come out to the last bit.

## Corners in the polyline

The simulator's centreline has a vertex every 0.1 m. Suppose instead you were handed the same road with a vertex only every 5 m. Two things change.

First, a shortcut that looked harmless stops working. Skip the segments: find the nearest vertex and use the distance to it as $|d|$. On the simulator's 0.1 m centreline its worst $|d|$ error over the lap is 0.00098 m, under a millimetre.

**Guess first:** with a vertex every 5 m, how far out can the shortcut be?

![The first bend of gentle with a vertex only every 5 m, zoomed in around the car at 43.30 s. Grey dots mark the vertices along the yellow line. The dashed grey line to the nearest vertex is long and slanted; the blue line at right angles to the nearest segment is short. The shortcut is 1.550 m out; the projection onto the segment is 0.039 m out.](figures/sparse_vertex.png)

**Answer:** by 1.550 m, at 43.30 s, with the car 1.27 m from the line. Projecting onto the 5 m segments is out by only 0.0391 m. The nearest vertex is not the nearest point. Midway between two vertices 5 m apart on a straight, the nearest vertex is 2.5 m along the road, so a car 1.8 m from the line is $\sqrt{1.8^2 + 2.5^2} = 3.081$ m from it, 1.281 m too far.

Second, even the segments miss. A straight segment between two points of a circle is a chord, and a chord cuts across the inside of the curve. The gap between a chord and its arc is largest at the chord's middle, and that largest gap is called the **sagitta**, read "SAJ-it-uh", Latin for arrow. For a chord of length $c$ on a circle of radius $R$ it is about

$$\begin{aligned}\text{sagitta} &\approx \frac{c^2}{8R}\end{aligned}$$

![A 5 m chord of gentle's 80 m first bend, with the sideways scale stretched so the gap shows. The yellow arc and the grey chord join two vertices 5 m apart. A blue brace at the chord's middle, labelled 0.0391 m, spans the gap between chord and arc: the chord's middle is 79.9609 m from the bend's centre instead of 80.](figures/sagitta.png)

<details>
<summary>How we got c² / (8R)</summary>

*Your first perception box* found that a circle of radius $R$ curves away from a straight line by about $d^2 / (2R)$ at $d$ metres ahead. In this lesson $d$ is sideways, so write $x$ for the distance ahead: the curve-away is $x^2 / (2R)$.

Stand at the middle of the arc, facing along it. The chord runs parallel to your direction, and its two ends are $c/2$ ahead and $c/2$ behind you. At $x = c/2$ the circle has curved away by

$$\begin{aligned}\frac{(c/2)^2}{2R} &= \frac{c^2}{8R}\end{aligned}$$

Both ends of the chord sit that far from your line, so the straight chord between them passes $c^2 / (8R)$ from the arc's middle. For $c = 5$ m and $R = 80$ m it is $25 / 640 = 0.0391$ m.

</details>

**Guess first:** at 1 m and 10 m spacing, will projecting onto segments still miss by exactly the sagitta?

![The largest error in d over a lap of gentle, against the spacing between vertices, 1, 5 and 10 m, on a scale where each grid line is ten times the one below. Grey dots, the nearest-vertex shortcut: 0.096, 1.550 and 3.924 m. Blue dots, projection onto segments: 0.0016, 0.0391 and 0.1562 m, each sitting on a grey curve labelled c squared over 8 times 80.](figures/spacing.png)

**Answer:** yes, to four decimal places. Projection misses by 0.0016, 0.0391 and 0.1562 m at 1, 5 and 10 m, and $c^2/(8 \times 80)$ gives the same three numbers. The formula is an approximation, but here it is good to four decimals. The shortcut is 61 times worse at 1 m, 40 times at 5 m and 25 times at 10 m.

For the simulator's own 0.1 m segments the sagitta is $0.1^2 / (8 \times 80) = 1.563 \times 10^{-5}$ m, 16 millionths of a metre. That is why straight segments are good enough to describe a round road. It is also the geometry behind the chord bias of *Your first perception box*: there the box's chord was metres long, so it cut far inside the bend.

![A vertex tie on gentle's first bend at 18.60 s, with the sideways scale stretched so the angle shows. Segments 1479 and 1480 meet at a shared vertex on the yellow line, and that vertex is the nearest point of the road to the car, which is 1.27 m to its right. Both segments' feet are the vertex. Their directions, 0.59920 and 0.60045 rad, differ by one corner angle, 0.00125 rad; the simulator keeps the first, segment 1479.](figures/vertex_tie.png)

One case is left. Sometimes the nearest point of the road is a vertex, shared by two segments, and both segments' feet are that same point. On the lap that happens at 20 of the 1778 steps, 1.1 %. $d$ is the same either way, but $\theta_{\text{road}}$ is the direction of whichever segment you pick, and neighbouring segments on `gentle`'s bends differ by one corner angle, $0.1 / 80 = 0.00125$ rad. The simulator keeps the first of the tied segments. The steps in this lesson, done in the same order, keep the same one, and match it exactly. A different but equally sensible tie-break, measured on the same lap, is 0.00125 rad off on 5 steps.

## Recap

- Sideways is measured from the road, not along a world axis. $y + 1.8$ is exact on the first straight and 163.6 m wrong on the far straight.
- Projecting onto a segment: $t$ is the forward part, clamped to the segment, and the signed distance to the foot takes the left part's sign. The nearest foot over all segments wins.
- The road frame $(s, d)$: $s$ is the arc length along the yellow line, $d$ the signed distance from it. $e = d + 1.8$ and $\psi = \text{wrap}(\theta - \theta_{\text{road}})$ reproduce the simulator's `lat` and `head_err` exactly.
- On a sparse polyline, use segments, not vertices; even segments miss a bend by about $c^2/(8R)$.

![A straight road from above, entered by a grey tunnel. Before the entrance the car's positions are grey dots, fixes from GPS. Inside the tunnel there are no dots, only a chain of short arrows joined end to end, each one step's move as the car itself measured it, 0.40 m long.](figures/next_lesson.png)

In a tunnel the GPS goes quiet. The car can still count its wheels' turns and read its own steering angle, so at every step it knows how far it moved and how much it turned since the last step, in its own frame. It has nothing else. If the car knows only how it moved since the last step, can it still know where it is?

That is the next lesson, *Poses you can chain*.

## Challenge problems

Optional. The first is hard; all three use only this lesson and the ones before it.

1. **The lane centre from the driver's seat.** On `gentle`'s first bend, at $s$ = 260 m, put the car 0.3 m left of its lane centre ($e = +0.3$) and pointing 0.04 rad left of the road ($\psi = +0.04$). Seen in the ego frame, how far to the left is the lane centre at $x$ metres ahead?
   - (a) Show that for small $x$ it is about $-e - \psi x + x^2/(2R)$, where $R$ = 81.8 m is the lane centre's radius. Use the curve-away from *Your first perception box* and a rotation by a small angle from *The view from the driver's seat*.
   - (b) The true values, from `world_to_ego` applied to the simulator's lane centre, are −0.3469, −0.0874, +1.3653 and +4.1277 m at $x$ = 5, 10, 20 and 30 m. Compute the quadratic and the straight line $-e - \psi x$ at each $x$. Roughly where is each first 0.1 m out?

![The setup for the first challenge problem: gentle's first bend from above around s = 260 m, with the yellow line, the dashed grey lane centre on its 81.8 m circle, and the car drawn as an arrow 0.3 m left of the lane centre, turned 0.04 rad to the left of the road. The ego frame's x and y axes are drawn in purple at the car's front axle.](figures/challenge_setup.png)

2. **Progress across the start line.** The last two steps of the lap have $s$ = 702.306 m and then $s$ = 0.045 m.
   - (a) What does the plain difference say the car moved in that step?
   - (b) Wrap the difference the way wrap brings an angle into one turn, but into $[-L/2, L/2)$, where $L$ = 702.655 m is the road's length to one more decimal. What do you get, and why is it a little less than the 0.40 m that one step at 8 m/s covers?
   - (c) How fast would the car have to go for this wrap to give the wrong answer?
3. **Why the shortcut is worst near the line.** On a straight road with a vertex every 5 m, put a point midway between two vertices, $|d|$ from the line. Show that the nearest-vertex shortcut is out by $\sqrt{d^2 + 2.5^2} - |d|$, and evaluate it at $|d|$ = 3.6, 1.8, 1.0 and 0.5 m. Is a car near the line or far from it hurt more?

<details>
<summary>Answers</summary>

1. (a) Stand at the lane-centre point level with the car, facing along the road. A point of the lane centre $x$ ahead along the road has curved $x^2/(2R)$ to the left, and the car is $e$ to the left of where you stand. So from a frame at the car pointing along the road, the lane centre $x$ ahead is about $x^2/(2R) - e$ to the left. The car's own frame is turned $\psi$ further left. *The view from the driver's seat* gives the left part of an arrow $(x, y)$ seen from a frame turned by $\psi$ as $y\cos\psi - x\sin\psi$, which for a small angle is about $y - \psi x$. Together: $-e - \psi x + x^2/(2R)$.

   (b) The quadratic gives −0.3472, −0.0888, +1.3450 and +4.0012 m, out by 0.0003, 0.0013, 0.0203 and 0.1265 m. The line gives −0.5, −0.7, −1.1 and −1.5 m, out by 0.1531, 0.6126, 2.4653 and 5.6277 m. The quadratic is first 0.1 m out at 28.49 m; the line already at 4.05 m. The line ignores the bend, and the bend's $x^2/(2R)$ grows fast.

2. (a) $0.045 - 702.306 = -702.261$ m: the car would have driven most of a lap backwards in 0.05 s.

   (b) Adding $L$ brings the difference into range: $-702.261 + 702.655 = +0.394$ m (+0.3936 m before rounding). It is less than 0.40 m because the car is on the second bend, on the outside of the yellow line, and $s$ is measured along the yellow line, which is the shorter circle.

   (c) The wrap gives the wrong answer only if one step moves more than $L/2$ = 351.3 m, which at 0.05 s per step is 7027 m/s.

3. The nearest vertex is 2.5 m along the road and $|d|$ across, so by Pythagoras it is $\sqrt{d^2 + 2.5^2}$ away, while the road is $|d|$ away. At 3.6, 1.8, 1.0 and 0.5 m the error is 0.783, 1.281, 1.693 and 2.050 m. A car near the line is hurt more: as $|d|$ shrinks toward 0 the error grows toward the whole 2.5 m.

</details>
