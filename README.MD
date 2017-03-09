# fuzzy C-Means with Tabu Search
## current Problem
1. ~~How to generate the neighbourhood with the local solution?~~
> F: Use random to generate a mat, plus the mat and the solution. This dot is in a hyperrectangle neighbouthood of the mat.

2. ~~How to define the tabu object, code or the mat?
if use the mat, what should we define the object, the mat itself or the area ?~~
> ASW: use mat and tabu the area of a rect around the mat(see mat as a dot in N-d space)

3. Calc the J first or judge the tabu-list first?

4. When get the minimum J, the FMI isn't the maximum. When the FMI is maximum, the J isn't the minimum. What should we choose?