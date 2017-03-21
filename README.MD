# fuzzy C-Means with Tabu Search
## current Problem
1. ~~How to generate the neighbourhood with the local solution?~~
> F: Use random to generate a mat, plus the mat and the solution. This dot is in a hyperrectangle neighbouthood of the mat.

2. ~~How to define the tabu object, code or the mat?
if use the mat, what should we define the object, the mat itself or the area ?~~
> F: use mat and tabu the area of a rect around the mat(see mat as a dot in N-d space)

3. Calc the J first or judge the tabu-list first?

4. When get the minimum J, the FMI isn't the maximum. When the FMI is maximum, the J isn't the minimum. What should we choose?

5. How to comprehend the saddle point in iteration seq?

6. ~~How to comprehend that the local minimun is related in the init value in iteration seq?~~
> F: In 3-d space ,z in the value of J, consider that there is a local minimun in the 2nd quadrant and a global minimun in the 4th quadrant. When y0>0, it's easy to find the local mininum and never find the global mininum.