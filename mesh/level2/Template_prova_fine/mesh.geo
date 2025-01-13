// Double wedge, with manual inserted refinement in zone of interest

defl_angle1 = 15 * 3.14/180;
defl_angle2 = 45 * 3.14/180;

length1 = 0.2;
length2 = 0.2;
length3 = 0.15;

f = 1.46;
h = 0.002 / f;
H = 0.1;

X1 = -0.05;
X3 = length1 * Cos(defl_angle1);
X4 = X3 + length2 * Cos(defl_angle2);
Y2 = length1 * Sin(defl_angle1);
Y4 = Y2 + length2 * Sin(defl_angle2);

X5 = X4 + length3;

// Points
Point(1)={X1, 0, 0, H};
Point(2)={0, 0, 0, h * 2};
Point(3)={X3, Y2, 0, h * 2};
Point(4)={X4, Y4, 0, h * 2};

Point(5)={X5, Y4, 0, h * 2};
Point(6)={X5, X5 - X1, 0, H};

Point(7)={X5, 0, 0, H};



// Construction points (for mesh size)
Point(8)={0.21177, 0.0822, 0, H};
Point(10)={X5, Y4+0.075, 0, H};
Point(11)={X5, Y4+0.115, 0, H};
Point(12)={0.213, 0.083, 0, H};
Point(13)={0.2468, 0.138, 0, H};
Point(14)={0.2792, 0.177, 0, H};
Point(15)={0.31485, 0.229, 0, H};
Point(16)={0.3324, 0.249, 0, H};
Point(17)={0.3595, 0.279, 0, H};
Point(18)={0.3943, 0.32, 0, H};
Point(19)={0.42825, 0.35, 0, H};
Point(20)={0.457, 0.378, 0, H};
Point(21)={0.4845, 0.405, 0, H};

Point(22)={X4 + 0.01, Y4 + 0.01, 0, H};
Point(23)={X4 + 0.01, Y4 + 0.005, 0, H};

Point(24)={X5, Y4 + 0.035, 0, H};
Point(25)={X5, Y4 + 0.1, 0, H};
Point(26)={X5, Y4 + 0.135, 0, H};
Point(27)={X5, Y4 + 0.15, 0, H};

Point(28)={X5, Y4 + 0.0525, 0, H};
Point(29)={X5, Y4 + 0.02, 0, H};
Point(30)={X5, Y4 + 0.175, 0, H};

Point(31)={X4, Y4 + 0.02, 0, H};
Point(32)={X5, Y4 + 0.185, 0, H};

Point(33)={0.105885, 0.0411, 0, H};

// Lines
Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,5};
Circle(5) = {1, 7, 6};
Line(6) = {5, 6};


// Construction lines (for mesh size)
Line(7) = {2, 33};
Line(8) = {33, 8};
Line(9) = {23, 10};
Line(10) = {22, 11};
Spline(11) = {3, 12:14}; // 2° shock part 1
Spline(12) = {14:17}; // 2° shock part 2
Spline(13) = {17:21}; // 2° shock part 3

Line(14) = {23, 24};
Line(15) = {23, 25};
Line(16) = {22, 26};
Line(17) = {22, 27};

Line(18) = {23, 28};

Line(19) = {23, 29};
Line(20) = {22, 30};

Line(21) = {31, 32};

// Build surfaces
Curve Loop(11)={1,2,3,4,6,-5};
Plane Surface(1) = {11};

// Fixing mesh size
Field[1] = Distance;
Field[1].CurvesList = {3, 4, 9, 10, 14, 15, 16, 17, 18, 19, 20, 21}; // expansion
Field[1].Sampling = 100;
Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = h * 2;
Field[2].SizeMax = H;
Field[2].DistMin = 0.01;
Field[2].DistMax = 0.1;

Field[3] = Distance;
Field[3].CurvesList = {7}; // first shock first part
Field[3].Sampling = 100;
Field[4] = Threshold;
Field[4].InField = 3;
Field[4].SizeMin = h / 3.4; 
Field[4].SizeMax = H;
Field[4].DistMin = 0.0035;
Field[4].DistMax = 0.3;

Field[5] = Distance;
Field[5].CurvesList = {8}; // first shock second part
Field[5].Sampling = 100;
Field[6] = Threshold;
Field[6].InField = 5;
Field[6].SizeMin = h / 3.6;
Field[6].SizeMax = H;
Field[6].DistMin = 0.0035;
Field[6].DistMax = 0.3;


Field[7] = Distance;
Field[7].CurvesList = {11}; // second shock first part
Field[7].Sampling = 100;
Field[8] = Threshold;
Field[8].InField = 7;
Field[8].SizeMin = h / 3.6;
Field[8].SizeMax = H;
Field[8].DistMin = 0.0085;
Field[8].DistMax = 0.3;

Field[9] = Distance;
Field[9].CurvesList = {12}; // second shock second part
Field[9].Sampling = 100;
Field[10] = Threshold;
Field[10].InField = 9;
Field[10].SizeMin = h / 2.6;
Field[10].SizeMax = H;
Field[10].DistMin = 0.011;
Field[10].DistMax = 0.3;

Field[11] = Distance;
Field[11].CurvesList = {13}; // second shock third part
Field[11].Sampling = 100;
Field[12] = Threshold;
Field[12].InField = 11;
Field[12].SizeMin = h / 1.9;
Field[12].SizeMax = H;
Field[12].DistMin = 0.017;
Field[12].DistMax = 0.3;

Field[13] = Min;
Field[13].FieldsList = {2, 4, 6, 8, 10, 12};
Background Field = 13;

// Physical names
Physical Surface("VOLUME")={1};
Physical Line("Symmetry")={1};
Physical Line("Wall")={2,3, 4};
Physical Line("Outlet")={6};
Physical Line("Farfield")={5};
