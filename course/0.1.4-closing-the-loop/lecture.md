# Closing the loop

Your perception box now hands the controller two numbers, offset and heading, twenty times a second. The provided controller turns them into a steering angle with two numbers of its own, and you have not looked inside it yet. How should two measurements become one steering command, and how hard should the car turn?

By the end of this lesson you will write that controller, `p_steer`, choose its two numbers yourself, and build a car whose perception and steering are both code you wrote. It is scored on `gentle`, and then it drives `curvy`, the road the provided car cannot finish.

![Two cars with the same perception box that differ only in the controller's two numbers, each road plotted as distance from the lane centre against distance along the road. Top, gentle: with the provided numbers, 0.35 and 2.7, the car rides up to 0.53 m left of the centre in the bends; with the numbers this lesson works out, 0.5 and 1.0, it never strays more than 0.021 m. Bottom, curvy: the provided numbers cross the yellow line at 48.3 m, on the first bend, while the new numbers finish the road and never stray more than 0.21 m.](figures/result_first.png)

Nothing changed between those two cars except two numbers. This lesson is about what they do: why the provided pair holds the car half a metre toward the inside of every bend on `gentle`, why the same pair sends it off `curvy`, and how to choose a better pair from measurements instead of by trial. On the way you meet the central idea of control, feedback.

## Open loop and closed loop

The simplest way to steer never looks at the road. Record the steering angle the provided car used at every step of its lap of `gentle`, then give the recording to a fresh car with its camera switched off and play it back, one angle per step. Steering by a plan, without measuring what happens, is called **open-loop** control.

The simulator has no randomness, so the replay drives exactly the same lap. Now make the plan very slightly wrong: add 0.002 rad to every recorded angle. That turns the front wheels 0.115° further left than planned, far too little to see.

**Guess first:** does the replayed car still finish the lap?

![Distance from the lane centre for the first 8.4 seconds on gentle, for two cars whose every steering angle is 0.002 rad too far left. The open-loop car, replaying a recording, drifts left faster and faster and reaches the yellow line, 1.8 m from the centre, after 8.40 s. The closed-loop car, the provided car steering from its camera with the same error added, never strays more than 0.0074 m from the centre over the same seconds.](figures/open_vs_closed.png)

**Answer:** no. It leaves the lane at 67.2 m, 8.40 s in, still on the first straight. Its error grows faster and faster: 0.04 m after 1 s, 0.13 m after 2 s, 0.44 m after 4 s and 1.65 m after 8 s. Wheels turned slightly left turn the car further left at every step, and a car pointing left drifts left. Nothing in a recording notices.

Add the same 0.002 rad to the provided car's commands while it steers from its camera, and it finishes the lap with a mean $|e|$ of 0.381 m instead of 0.376 m. It measures where it is, and the measurement changes the next steering angle, which changes where the car goes, and so the next measurement. Steering from measurements of the result is called **closed-loop** control, and the path from the result back into the command is called **feedback**. Karl Åström and Richard Murray's textbook on control defines it:

> "The term feedback refers to a situation in which two (or more) dynamical systems are connected together such that each system influences the other" (§1.1)

A dynamical system is anything whose state changes over time. Here they are the car and its controller: where the car is decides the steering, and the steering decides where the car goes next.

![Two loops, the open loop above and the closed loop below. Open loop: a recorded list of steering angles goes into the car, and nothing comes back. Closed loop: the camera picture goes to the perception box, its offset and heading go to the controller box, the steering angle goes to the car, and the car's new position changes the next camera picture.](figures/loop_kinds.png)

## Steering in proportion to the error

The controller box receives the perception box's three answers: the offset (m), the heading ψ (rad), both left positive, and `valid`. It returns one number, the **steering angle** δ (read "delta"): how far the front wheels are turned from the car's own direction, in radians, left positive. The simulator's wheels turn at most 0.5 rad, and at most 0.05 rad per 0.05 s step.

Start with the offset alone. A car 0.4 m left of the lane centre should steer right, and the further off it is, the harder. Stanley, the robot car from *Meet your car*, says it (the trajectory is the path to follow): "The larger this error, the stronger the steering response toward the trajectory." (Thrun et al., 2006, p. 684). The simplest rule with that property multiplies the offset by a fixed number:

$$\color{#83C167}{\delta} = -k_{\text{off}} \cdot \text{offset}$$

$k_{\text{off}}$, read "k off", is a **gain**: radians of steering per metre of offset. The provided car's is 0.35, so at 0.4 m left, $\delta = -0.35 \times 0.4 = -0.14$ rad, a turn to the right.

The minus sign makes every correction oppose the error, called **negative feedback**. Flip a sign and you get positive feedback, the loop of *Your first perception box* in which each correction feeds the error: flipping only the offset's sign left `gentle` after 39.9 m, 5.00 s; flipping the whole rule, after 8.2 m, 1.10 s. A command equal to a fixed gain times the error is called **proportional control**, and a controller built on it is a **P controller**.

Give the car a push on `gentle`'s first straight: hold the wheels at 0.025 rad left for 1 s, then hand over to the offset-only rule.

**Guess first:** when the car gets back to the lane centre, does it stop there?

![A push test on gentle's first straight, plotted as distance from the lane centre against time. The wheels are held at 0.025 rad left for 1 s, then the offset-only rule with k_off 0.35 takes over. The car is 0.50 m left at the handover, comes back, crosses the centre at 1.65 s pointing 0.082 rad to the right, and swings 0.135 m past it before it settles.](figures/offset_only.png)

**Answer:** no. At the handover the car is 0.50 m left, pointing 0.074 rad left. The rule steers it back, but at the centre the offset is zero, so the rule straightens the wheels while the car still points 0.082 rad right. A car pointing right keeps moving right: it swings 0.135 m past the centre.

The rule ignored the heading. A car pointing ψ away from the lane's direction drifts sideways at $v \sin\psi$, where $v$ is its speed, 8 m/s, and $\sin\psi$ is the sideways share of that speed when the car points ψ off the lane. At 0.05 rad that is 0.400 m every second. A heading is an offset in the making, so the full rule also steers against it, with a gain of its own, $k_{\text{head}}$ ("k head"):

$$\begin{aligned}\color{#83C167}{\delta} &= -k_{\text{off}} \cdot \text{offset}\\ &\quad - k_{\text{head}} \cdot \color{#D147BD}{\psi}\end{aligned}$$

The provided $k_{\text{head}}$ is 2.7. A car pointing 0.05 rad left gets $-2.7 \times 0.05 = -0.135$ rad from the heading term. A car 0.4 m left **and** pointing 0.05 rad left gets both terms, $-0.14 - 0.135 = -0.275$ rad.

![The steering rule as a graph: the steering angle it returns against the offset, for a heading of 0 and for a heading of 0.05 rad left, with the provided gains. Each line falls by 0.35 rad for every metre of offset, and the heading lowers the whole line by 0.135 rad. At 0.4 m left the two lines give −0.14 rad and −0.275 rad. Both lines go flat at +0.5 rad and −0.5 rad, the furthest the wheels can turn.](figures/p_law.png)

Two more rules finish the controller.

- **Clip.** A car 1.5 m left and pointing 0.1 rad left gets −0.795 rad, but the wheels stop at −0.5 rad. So the controller **clips** its answer: anything above +0.5 becomes +0.5, and anything below −0.5 becomes −0.5. With the heading at 0, the offset alone reaches the clip at 1.43 m.
- **Invalid.** When `valid` is False the box could not find both lines, and its numbers are NaN, "not a number". The controller returns 0.0, wheels straight, as the provided one does.

That is the lab's Exercise A, `p_steer(est, k_off, k_head)`.

## How the heading gain shapes the return

Repeat the push with the full rule, $k_{\text{off}} = 0.35$ and six heading gains: 0, 0.5, 1, 2, 2.7 (the provided one) and 5.

**Guess first:** which heading gain brings the car within 5 cm of the centre fastest without swinging past it?

![The same push test for six heading gains, all with k_off 0.35, plotted as distance from the lane centre against time, each gain a darker shade of blue than the last. Heading gain 0 swings 0.135 m past the centre, and 0.5 swings 0.033 m past. Heading gain 1 comes back without swinging past, never more than about 0.001 m beyond the centre, and is within 5 cm of the centre 0.95 s after the handover. Larger gains come back more and more slowly: 1.75 s for 2, 2.30 s for 2.7 and 3.75 s for 5. Gain 5 first dips to 0.064 m at 1.50 s; the wheel, turning at most 0.05 rad per step, then overshoots and swings the front axle back out to 0.25 m at 1.80 s. The 5 cm count is from the last time each curve leaves the band.](figures/push_family.png)

**Answer:** 1. It is within 5 cm 0.95 s after the handover and never swings more than about 1 mm past the centre. Below 1 the car swings past: 0.5 reaches the 5 cm band sooner, after 0.70 s, but only by crossing the centre, by 0.033 m. Above 1 the car crawls back. As soon as it turns toward the centre it points right, and the heading term then steers left, away from the centre, the harder the bigger the gain.

Why exactly 1? Write the rule with $k_{\text{head}} = 1$, then move the heading to the left-hand side:

$$\begin{aligned}\color{#83C167}{\delta} &= -k_{\text{off}} \cdot \text{offset}\\ &\quad - \color{#D147BD}{\psi}\\ \color{#D147BD}{\psi} + \color{#83C167}{\delta} &=\\ &\quad -k_{\text{off}} \cdot \text{offset}\end{aligned}$$

ψ is the car's direction measured from the lane's direction, and δ is the wheels' direction measured from the car's. Their sum is the wheels' direction measured from the lane's. So with a heading gain of 1 the rule sets where the front wheels point relative to the lane: back toward the centre, by 0.35 rad for each metre of offset, whatever the car's body is doing.

![A car seen from above, 0.50 m left of the lane centre and pointing 0.074 rad left of the lane's direction. The angle from the lane's direction to the body is the heading ψ, the angle from the body to the front wheels is the steering angle δ, and the angle from the lane's direction to the front wheels is ψ plus δ. With a heading gain of 1 the rule sets ψ plus δ to −0.17 rad here, aiming the wheels back toward the lane centre. The arrow along the bottom is the offset e, 0.50 m, from the front axle to the lane centre. Angles are drawn 3 times larger than they are.](figures/wheel_vs_lane.png)

The front axle, where $e$ is measured, rolls the way its wheels point. It heads back steeply when it is far off and more gently as the offset shrinks, so it eases onto the centre without crossing it. At the handover the rule asks for −0.248 rad of steering. With the car's heading at +0.074 rad, that points the wheels −0.17 rad from the lane's direction: 0.35 rad back for each metre of the 0.50 m offset.

<details>
<summary>Optional: how we got about 0.95 s</summary>

On a straight the box's offset is within a few millimetres of $e$, so treat them as equal. The front axle moves sideways at $v \sin(\psi + \delta)$, and for small angles $\sin x \approx x$. With a heading gain of 1 that sideways speed is $-v\,k_{\text{off}}\,e$. Write $e(t)$, read "e at time t", and $e_0$ for its value at the handover: the offset shrinks at a rate proportional to itself, losing $v\,k_{\text{off}} = 2.8$ times its own size per second. A quantity that shrinks like this falls by the same factor in every equal stretch of time. It is called an exponential decay, and it never overshoots zero. In the equation, $\mathrm{e}$ is the number 2.718, not the lateral error, and $\ln$ is the natural logarithm, which undoes it.

$$\begin{aligned}e(t) &= e_0\, \mathrm{e}^{-2.8\,t}\\ t_{5\,\text{cm}} &= \frac{\ln(0.496/0.05)}{2.8}\\ &= 0.82\ \text{s}\end{aligned}$$

The measured return takes 0.95 s. Part of the difference is the wheels: they cannot jump from +0.025 rad to −0.248 rad, and at 0.05 rad per step they need 6 steps to get there.

</details>

## Holding a bend

Now a bend. To watch the controller on its own, use the perfect box of *Your first perception box*, which reads the true $e$ and ψ straight from the simulator. Your car can never have one. Even with a perfect box, a car with the provided gains rides 0.160 m toward the inside of each of `gentle`'s bends.

To hold a bend the wheels must stay turned, and wheels held at a fixed angle δ drive a circle. Its radius $R$, measured to the front axle, has $\sin\delta = L/R$, where $L$ is the 2.7 m wheelbase from *Meet your car*. On `gentle`'s bends your lane's centre is a circle of radius 81.8 m, so holding it takes δ = 0.0330 rad. For angles this small, $\sin\delta \approx \delta$, so $\delta \approx L/R$.

![A car seen from above holding a left bend with its front wheels turned by δ. The rear wheel rolls straight ahead and the front wheel rolls at δ, so the car turns about the point where the lines drawn at right angles to both wheels meet. The line from that point to the front axle is the radius R, the wheelbase L is the side opposite the angle δ, and so sin δ equals L divided by R. Angles are drawn larger than they are.](figures/circle.png)

<details>
<summary>How we got sin δ = L / R</summary>

The rear wheel cannot slide sideways, so it moves around a turning point on the line through the rear axle at right angles to the car. The front wheel cannot slide sideways either, so the same point is on the line through the front axle at right angles to the front wheel. The two lines meet at the angle δ, because each is turned from the other by the same angle as the wheels. The rear axle, the front axle and the turning point make a triangle with a right angle at the rear axle. Its side opposite δ is the wheelbase $L$, and its longest side is $R$, from the turning point to the front axle. So $\sin\delta = L/R$.

To get the angle back from its sine, use arcsin, read "arc-sine", which undoes sin the way arctan undoes tan. For `gentle`, $\arcsin(2.7/81.8) = 0.0330$ rad. Holding the simulator's wheels at that angle traces a front-axle circle of 81.800 m.

</details>

Holding the bend at a steady offset, the front axle moves along the lane, so its wheels point along the lane: $\psi + \delta = 0$, and so $\psi = -\delta$. The car's body points 0.033 rad toward the outside of the bend. With a perfect box the measured heading in the bends is −0.0332 rad.

**Guess first:** with a perfect box and $k_{\text{off}} = 0.35$, which heading gain keeps the car on the lane centre through `gentle`'s bends: 0, 1 or 2.7?

![One lap of gentle with a perfect box and k_off 0.35, plotted as distance from the lane centre against distance along the road, for three heading gains in three shades of blue. Through both bends, heading gain 2.7 rides 0.160 m toward the inside, heading gain 0 rides 0.094 m toward the outside, and heading gain 1 stays within 0.003 m of the centre the whole lap. The bends are shaded.](figures/steady_state.png)

**Answer:** 1. Put $\psi = -\delta$ into the rule and solve for $e$:

$$\begin{aligned}\color{#83C167}{\delta} &= -k_{\text{off}}\,\color{#58C4DD}{e}\\ &\quad - k_{\text{head}}\,\color{#D147BD}{\psi}\\ \color{#83C167}{\delta} &= -k_{\text{off}}\,\color{#58C4DD}{e}\\ &\quad + k_{\text{head}}\,\color{#83C167}{\delta}\\ k_{\text{off}}\,\color{#58C4DD}{e} &= (k_{\text{head}} - 1)\,\color{#83C167}{\delta}\end{aligned}$$

So the car settles at $e = (k_{\text{head}} - 1)\,\delta / k_{\text{off}}$:

- Heading gain 2.7: $1.7 \times 0.0330 / 0.35 = +0.160$ m. Measured: +0.160 m.
- Heading gain 1: 0 m. Measured: −0.0001 m.
- Heading gain 0: $-0.0330 / 0.35 = -0.094$ m, toward the outside. Measured: −0.094 m.

In words: with ψ = −δ, the heading term supplies $k_{\text{head}}\,\delta$ of steering. A gain of 1 supplies exactly the δ the bend needs. A gain of 2.7 supplies 1.7 δ too much, so the car sits inside the bend, where the offset term takes the extra back.

Åström and Murray name this weakness: "proportional control has the drawback that the process variable often deviates from its reference value" (§1.6). The process variable is the quantity being controlled, here $e$, and its reference value is the one you want, 0.

![One run of curvy with a perfect box and k_off 0.35, plotted as distance from the lane centre against distance along the road, for heading gains 2.7 and 1, with the left bends in a lighter shade and the right bends in a darker one, each labelled. With 2.7 the car sits toward the inside of every bend, left on the left bends and right on the right ones, up to 0.674 m from the centre; the formula predicts +0.603 m on the left bends, which the car reaches, and −0.723 m on the right ones, where it gets only to −0.674 m. With 1 it stays within 0.023 m.](figures/steady_curvy.png)

`curvy`'s left bends are 3.8 times tighter than `gentle`'s and its right bends 4.5 times (radius 18.2 m). On the left bends the car reaches the predicted +0.603 m. On the right bends it gets only to −0.674 m of the −0.723 m predicted, because it enters each right bend from about +0.6 m on the left, and the bend, 3.93 s long, ends first. Stanley's steering law has the heading in it with a gain of exactly 1. Its Eq. 7, where "(t)" means "at time t" and $k$ is a gain, is

$$\begin{aligned}\delta(t) &= \psi(t)\\ &\quad + \arctan\frac{k\,x(t)}{u(t)}\end{aligned}$$

<details>
<summary>How we read Stanley's symbols</summary>

Here $x(t)$ is Stanley's name for the lateral error, the cross-track error, and $u(t)$ is the speed. Its ψ is the lane's direction measured from the car, which is minus your heading (p. 684). You derive the whole law in *Lateral control*.

</details>

Read §9.2 of the Stanley paper (Thrun et al., 2006) now.

## Where the box pulls the car

Back to the real box. In `gentle`'s bends the provided car rides 0.530 m toward the inside, not 0.160 m. The difference is the chord bias of *Your first perception box*: the chord through the box's lane-centre points runs to the car's right, so the box puts the car too far left and pointing too far right. Over the middles of `gentle`'s bends the box reports the provided car's offset 0.128 m too far left and its heading 0.064 rad too far right (*Your first perception box* quotes 0.129 m and 0.0645 rad from 40 frames). Call these $b_{\text{off}} = +0.128$ m and $b_{\text{head}} = -0.064$ rad.

The controller cannot tell. Redo the bend calculation with it steering on $e + b_{\text{off}}$ and $\psi + b_{\text{head}}$, and the car settles at

$$\begin{aligned}\color{#58C4DD}{e} &= \frac{(k_{\text{head}} - 1)\,\color{#83C167}{\delta}}{k_{\text{off}}}\\ &\quad - b_{\text{off}}\\ &\quad - \frac{k_{\text{head}}}{k_{\text{off}}}\, b_{\text{head}}\end{aligned}$$

That is three pieces: the P controller's own bend offset, the offset bias, and the heading bias multiplied by $k_{\text{head}} / k_{\text{off}}$.

**Guess first:** with the provided gains, which piece is the biggest?

![The provided car's bend offset on gentle, built from three pieces as bars. The P controller's own bend offset is +0.160 m. The box's offset bias moves the car 0.128 m toward the outside. The box's heading bias, multiplied by 2.7 divided by 0.35, which is 7.71 m, moves it 0.497 m toward the inside. Each bar starts where the one above ended. Together they predict +0.529 m, next to the measured +0.530 m.](figures/bias_terms.png)

**Answer:** the heading bias, by far. The controller multiplies the heading by 2.7 and the offset by only 0.35, so in a bend each radian of heading error counts as much as 7.71 m of offset error, and the box's 0.064 rad becomes 0.497 m. The offset bias alone would move the car 0.128 m toward the outside, because the box reports it further left than it is. The two pull opposite ways, and the heading bias wins.

<details>
<summary>How we got the formula with biases</summary>

The controller steers on the box's numbers:

$$\begin{aligned}\delta &= -k_{\text{off}}\,(e + b_{\text{off}})\\ &\quad - k_{\text{head}}\,(\psi + b_{\text{head}})\end{aligned}$$

Holding the bend still needs $\psi = -\delta$. Put it in, collect the $\delta$ terms on one side and divide by $k_{\text{off}}$:

$$\begin{aligned}k_{\text{off}}\,e &= (k_{\text{head}} - 1)\,\delta\\ &\quad - k_{\text{off}}\,b_{\text{off}}\\ &\quad - k_{\text{head}}\,b_{\text{head}}\end{aligned}$$

With the provided gains: $+0.160 - 0.128 + 0.497 = +0.529$ m.

</details>

## Why the provided car leaves curvy

On `curvy`'s left bends your lane's centre has a radius of 21.8 m, so holding one takes δ = 0.124 rad, and the chord bias grows too. Park a car on the lane centre in the middle of the first left bend, pointing the way it would to hold the bend. The box says it is 0.558 m left of centre, and its heading is 0.26 rad too far right.

Put those numbers into the formula with the provided gains: +0.603 m from the P controller, −0.558 m from the offset bias and +1.994 m from the heading bias, +2.039 m in all. The yellow line is 1.8 m from the centre. The steady bend offset this controller aims for lies beyond the paint.

![The provided car on curvy, from the start until it leaves the lane, plotted against distance along the road. Top: the true distance from the lane centre in blue and the box's offset in coral, with the first left bend starting at 30 m. At 30.3 m the car is already 0.555 m left while the box says 1.041 m, and the car reaches the yellow line at 48.3 m. Bottom: the true heading in pink and the box's heading in coral; at 44.3 m the car points 0.085 rad right and the box says 0.342 rad right.](figures/curvy_leave.png)

The box's two lane-centre points are about 2.6 m and 7.2 m ahead of the car, so it starts to see the bend before the car reaches it. At 30.3 m, just past the bend's start at 30.0 m, the car is already 0.555 m left.

**Worked example.** At 44.3 m the car is truly 1.701 m left and pointing 0.0848 rad right. On those true numbers the rule would steer −0.366 rad, a hard right back toward the centre. The box says 2.321 m left and pointing 0.3416 rad right, and on those numbers the rule steers +0.110 rad, left, further left, toward the inside of the bend. Four metres later the car crosses the yellow line.

<details>
<summary>How we got −0.366 rad and +0.110 rad</summary>

$$\begin{aligned}&\text{truth:}\\ &-(0.35 \times 1.701\\ &\;+ 2.7 \times (-0.0848))\\ &= -0.366\\ &\text{box:}\\ &-(0.35 \times 2.321\\ &\;+ 2.7 \times (-0.3416))\\ &= +0.110\end{aligned}$$

On the box's numbers the heading term, $+2.7 \times 0.3416$, outweighs the offset term, $-0.35 \times 2.321$.

</details>

## Tuning your car

You have two numbers to choose, and the bias formula says what each one does. On `gentle` its two bias pieces are −0.128 m and $(k_{\text{head}}/k_{\text{off}}) \times 0.064$. They cancel when $k_{\text{head}} / k_{\text{off}} = 0.128 / 0.064$, which is 2.0 m. The provided car's ratio is 7.71 m.

<details>
<summary>Optional: why the ratio is about 2 m on every bend</summary>

Take a car on the lane centre of a bend of radius $R$, with the box's two lane-centre points $d_1$ and $d_2$ ahead, as in the chord formula of *Your first perception box*. On this bend $d_1 = 2.79$ m and $d_2 = 7.35$ m, and for small angles the chord's offset is $d_1 d_2 / (2R)$ to the car's right, where it meets the car's axis, and its heading is $(d_1 + d_2)/(2R)$, tilted left. Divide one by the other and $R$ cancels:

$$\begin{aligned}\frac{b_{\text{off}}}{|b_{\text{head}}|} &= \frac{d_1 d_2}{d_1 + d_2}\\ &= \frac{2.79 \times 7.35}{2.79 + 7.35}\\ &= 2.02\ \text{m}\end{aligned}$$

On `gentle` the box's measured ratio is 1.995 m. Parked on `curvy`'s first left bend it is 2.16 m: close, not exact. The steering a bend needs, $\delta \approx L/R$, also shrinks as $1/R$, so the gains that give no bend offset hardly depend on the bend at all.

</details>

A heading gain of 1 also removes the P controller's own piece. Take both together: $k_{\text{head}} = 1$, and so $k_{\text{off}} = 0.5$. The formula predicts a bend offset on `gentle` of +0.0003 m. The car measures −0.003 m.

**Guess first:** the maps below show mean $|e|$ for 91 pairs of gains on each road. Is the best pair for `gentle` also the best for `curvy`?

![Two maps, gentle on top and curvy below, each a grid of 13 offset gains across and 7 heading gains up, coloured by mean |e| from pale blue for small to dark blue for large, with cars that left the lane hatched grey. A dashed line on each map marks the pairs for which the bias formula predicts no bend offset on that road; the two lines nearly coincide, and both pass through or next to 0.5 and 1.0. Marked: the provided car at 0.35 and 2.7, which leaves curvy; the 2× car from Meet your car at 0.7 and 2.7, mean |e| 0.502 m on curvy; the 4× car at 1.4 and 2.7, 0.087 m on curvy; and 0.5 and 1.0, the best on gentle at 0.003 m and 0.083 m on curvy.](figures/tuning_map.png)

**Answer:** not quite. On `gentle` the best pair on the map is 0.5 and 1.0, at 0.003 m. On `curvy` the best is 0.7 and 1.5, at 0.066 m, and 0.5 and 1.0 scores 0.083 m. Each road's best cars lie close to its dashed line, and the two lines nearly coincide: a tighter bend needs more steering, but its chord biases grow in the same proportion. On `curvy` the formula is rougher. For 0.5 and 1.0 it predicts −0.041 m in the middle of the first left bend, and the car sits at −0.075 m.

The 4× car from *Meet your car*, 1.4 and 2.7, lies near the lines, and it stayed within 0.35 m on `curvy`. The 2× car, 0.7 and 2.7, lies far above them and swung up to 0.82 m.

**A patch, not a fix.** These gains work because two errors of the box cancel. The box is still wrong by 0.128 m and 0.064 rad in every `gentle` bend, and a different box would need different gains. The real fixes are a box that fits a curve instead of a chord, in *Classical lane perception*, and a controller that aims at the road ahead, in *Lateral control*.

### Too much gain

Why not turn the gains up further? Keep $k_{\text{head}} = 2.7$ and raise $k_{\text{off}}$. On `gentle`, mean $|e|$ falls from 0.376 m at 0.35 to 0.014 m at 2.0, then rises: 0.042 m at 3.0 and 0.057 m at 4.0. At 5.0 the car leaves the lane late in the lap, and at 6.0 it leaves after only 33.8 m, 4.30 s in, on the first straight.

**Guess first:** what makes the 6.0 car crash on a straight road?

![Distance from the lane centre for the first 20 s on gentle with heading gain 2.7, for offset gains 0.05, 0.35, 1.4 and 6.0 in shades of blue. 0.05 drifts toward the inside of the first bend and leaves at 15.70 s; 0.35 rides up to 0.531 m from the centre; 1.4 stays within 0.041 m; 6.0 swings wider and wider and leaves at 4.30 s.](figures/gain_sweep.png)

**Answer:** the wheels' speed. They turn at most 0.05 rad per step, and at $k_{\text{off}} = 6$ a few centimetres of offset ask for a big change of steering. One second in, the car is 0.041 m right of centre, the rule asks for +0.315 rad, and the wheels are still at −0.027 rad. By the time they have turned left, the car has crossed the centre and the rule wants the opposite: at 3 s the car is 0.148 m left, the rule asks for −0.5 rad, and the wheels are at +0.216 rad. The wheels turn as fast as they can in 80% of the steps, always late, and each swing is wider than the last.

![The first 4.3 s of the offset gain 6.0 car on gentle, plotted as steering angle in radians against time. The steering angle the rule asks for, in light green, swings between −0.5 and +0.5 rad. The angle the wheels actually reach, in dark green, always lags behind it and never gets near the extremes. Dotted vertical lines mark 1 s and 3 s.](figures/wheel_lag.png)

With the wheel-speed limit lifted, the same gains finish `gentle` with a mean $|e|$ of 0.107 m. But the wheels then swing back and forth at about 19 rad/s, which no real steering can do. The car's limits set the top of the useful range of gains.

## Your whole car

Your last step builds the car:

```python
my_car = Car(perception=estimate_lane, controller=my_controller)
```

`estimate_lane` is your box from *Your first perception box*, and `my_controller(est, obs)` returns `p_steer(est, K_OFF, K_HEAD)`; `obs` is the same camera observation the box received, and your controller may ignore it. The `Car` pairs your steering angle with the provided speed controller, which holds 8 m/s. You replace that one in *Longitudinal control and vehicle motion*.

![The loop your car now runs every 0.05 s: the camera picture, then paint_masks from Images are arrays, then estimate_lane from Your first perception box, then p_steer from Closing the loop, then the car moves. Under each box is its known weakness and the lesson that fixes it: the chord bias on bends, Classical lane perception; steering only on the error at the car, Lateral control; and the provided speed controller, Longitudinal control and vehicle motion. The perception box and the steering are yours; paint_masks and the speed controller are the provided ones.](figures/loop_lookback.png)

The lab's Exercise B checks that `my_car` uses neither provided box and finishes `gentle` with a mean $|e|$ below 0.5 m. Then it drives `curvy`, ungraded. The lab starts `K_OFF` and `K_HEAD` at the provided 0.35 and 2.7.

**Guess first:** your box is not the provided one. With the provided gains, will your car leave `curvy` before 48.3 m, at it, or after?

**Answer**, measured with a box that works the way the provided one does: with the provided gains it leaves at 48.3 m, exactly like the provided car. With 0.5 and 1.0 it finishes `curvy` at a mean $|e|$ of 0.083 m, and `gentle` at 0.003 m. **Approximation:** the lab of *Your first perception box* accepts a box that lands within 0.25 m and 0.05 rad of the truth on 9 test poses in 10, so your own box's numbers may land somewhere else.

The perception box and the steering are now yours, and each has a weakness you have measured.

## A 1988 car that learned to steer

Your car has a hand-written perception box and a hand-written controller. In 1988, Dean Pomerleau at Carnegie Mellon University replaced both with one small neural network, ALVINN: "a 3-layer back-propagation network designed for the task of road following" (p. 305). You met its camera grid in *Images are arrays*. Back-propagation is the method that trains such a network from examples, and you meet it in *Neural nets and CNNs in PyTorch*.

![ALVINN's network as the paper describes it. On the left, 1217 inputs: a 30 by 32 video grid of 960 units, an 8 by 32 laser range grid of 256 units and one feedback unit. In the middle, 29 hidden units. On the right, 46 output units: 45 each standing for one steering direction, from sharp left through straight ahead to sharp right, and one road-brightness feedback unit.](figures/alvinn.png)

- **Inputs:** the 30 by 32 video grid fed the blue colour band, an 8 by 32 grid from a laser range finder, each unit's activation set by how close that patch is, and one feedback unit: 1217 inputs, into 29 hidden units, the layer between the inputs and the outputs.
- **Outputs:** 46 units: 45 direction units, from sharp left to sharp right, the middle one meaning "travel straight ahead" (p. 306), and one road-brightness feedback unit.
- **Training:** 1200 simulated road snapshots (p. 308).
- **Result:** it drove the Navlab "at a speed of 1/2 meter per second along a 400 meter path through a wooded area" (pp. 308–309). CMU's traditional image-processing road follower, on the Warp computer, reached similar accuracy at 1 m/s; ALVINN ran on an on-board Sun.

<details>
<summary>How we got 1217 inputs</summary>

Each grid has one unit per cell, so its size is rows times columns: $30 \times 32 = 960$ and $8 \times 32 = 256$. Add the one feedback unit:

$$960 + 256 + 1 = 1217$$

</details>

Read the paper's Figure 1 and "Network Architecture", then answer:

1. Your controller steers on the error at the front axle. What does ALVINN's training target aim at instead?
2. Where in ALVINN is the perception box, and where is the controller?
3. `p_steer` returns one angle. ALVINN returns 46 numbers. What can go wrong where the road forks?

<details>
<summary>Answers</summary>

1. A point ahead: its target is the curvature, how sharply the path bends, "which would bring the vehicle to the road center 7 meters ahead of its current position" (p. 306). Aiming at a point ahead is called lookahead, and it is the controller idea of *Lateral control*.
2. Nowhere separately. The inputs are pixels and proximity, the outputs are steering directions, and no unit in between means offset or heading.
3. The most active of the 45 direction units gives the steering direction. At a fork the network can light up two directions, one for each branch, and "The result is often an oscillation" (p. 312).

</details>

## Recap

- Feedback measures the result and corrects it. A recorded plan 0.002 rad off leaves `gentle` after 8.40 s; the provided car with the same error finishes.
- The P controller steers $-(k_{\text{off}} \cdot \text{offset} + k_{\text{head}}\,\psi)$, clipped to ±0.5 rad, straight when invalid. A heading gain of 1 aims the wheels at the lane and holds a bend with no offset.
- On a bend the box's chord bias moves the car by $-b_{\text{off}} - (k_{\text{head}}/k_{\text{off}})\,b_{\text{head}}$. Gains with $k_{\text{head}} \approx 2\,k_{\text{off}}$ cancel it for this box: a patch, not a fix.

![One camera picture on gentle's first straight with the box's two lane-centre points marked, with the two points labelled outside the picture, and beside it the same two points drawn from above, in metres forward and left of the car. A third mark, a ring with a question mark on the lane centre 10 m ahead, beyond both, appears only in the view from above.](figures/next_lesson.png)

ALVINN aimed at the road centre 7 m ahead, and the question mark marks a point of that kind, 10 m ahead. To aim at a point ahead, a controller must say where that point is, and the answer depends on who measures it: the camera counts in pixels, the car in metres forward and to its left, the road in metres along it and across it. Each way of measuring position is called a frame of reference, or frame, a different meaning from a camera frame. How do you move a point from one frame to another?

That is the next module, *Frames and transforms*, which opens with *Positions and headings*.

## Challenge problems

Optional and hard, using only this lesson.

1. **Predict the crash.** The open-loop car's wheels are $b = 0.002$ rad off. That moves the front axle sideways at $v\,b$ and turns the car's heading at $v\,b/L$ radians per second, which adds sideways speed too. Write $e(t)$ and find when it reaches 1.8 m.
2. **The fastest stable gain.** With a perfect box, a heading gain of 1 and no limit on the wheels' speed, each 0.05 s step moves the front axle 0.4 m forward. Show that each step multiplies the offset by about $1 - 0.4\,k_{\text{off}}$. Above which gain does it stop shrinking?
3. **Two predictions.** Use the bias formula with the `gentle` biases to predict the bend offsets of the 2× car, 0.7 and 2.7, and of 1.5 and 2.7.

<details>
<summary>Answers</summary>

1. $e(t) = v\,b\,t + v^2 b\,t^2 / (2L)$. Setting it to 1.8 m gives $t = 8.38$ s, against the measured 8.40 s.
2. With a gain of 1 the wheels point $-k_{\text{off}}\,e$ from the lane, so one step moves the front axle $0.4 \times k_{\text{off}}\,e$ back toward the centre. The offset stops shrinking once $|1 - 0.4\,k_{\text{off}}| \ge 1$, which is $k_{\text{off}} \ge 5$. Measured: 4.5 settles to a mean $|e|$ of 0.000 m, while 5.0 and 5.5 chatter from side to side at 0.078 m and 0.079 m.
3. For 0.7 and 2.7 the formula predicts +0.200 m, and the car measures +0.199 m. For 1.5 and 2.7 it predicts +0.025 m, and the car measures +0.023 m.

</details>
