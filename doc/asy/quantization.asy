size(3inch);
import settings;
settings.render=10;
import three;
import graph3;

currentprojection=orthographic(5,4,2);
currentlight=White;

real x = .525731112119133606;
real z = .850650808352039932;

triple [] IcoPoints = 
{
	( 0, x, z),
	( 0, -x, z),
	( z, 0, x),
	( x, z, 0),
	( -x, z, 0),
	(-z, 0, x),
	( z, 0, -x),
	( 0, x, -z),
	(-z, 0, -x),
	( -x, -z, 0),
	( x, -z, 0),
	( 0, -x, -z)
};

// Faces listed as triples (i,j,k) corresponding
// to the face through IcoPoints[i], IcoPoints[j] and IcoPoints[k].
triple [] IcoFaces = 
{
	// upper cap
	(0,1,2), (0,2,3), (0,3,4), (0,4,5), (0,5,1),
	// upper band
	(11,6,7), (11,7,8), (11,8,9), (11,9,10), (11,10,6),
	// lower band
	(10,1,2), (6,2,3), (7,3,4), (8,4,5), (9,5,1),
	// lower cap
	(3,6,7), (4,7,8), (5,8,9), (1,9,10), (2,10,6)
};

void draw_face(triple face, pen color) 
{
	int i=round(face.x),
	    j=round(face.y),
	    k=round(face.z);
  
	draw(IcoPoints[i]--IcoPoints[j]--IcoPoints[k]--cycle, color);
}

int selectedIndex = 18;

for(int k = 0; k < 20; ++k)
{
	draw_face(IcoFaces[k], lightgrey);
}

triple face = IcoFaces[selectedIndex];


pen qPen = heavygreen;
pen nPen = royalblue;

draw(surface(IcoPoints[round(face.x)]--IcoPoints[round(face.y)]--IcoPoints[round(face.z)]--cycle), grey+opacity(0.2));

triple center = unit(IcoPoints[round(face.x)] + 
                 IcoPoints[round(face.y)] + 
                 IcoPoints[round(face.z)]);

triple q = center * 1.25;
triple n = rotate(-25, X) * q;

draw((0,0,0)..IcoPoints[round(face.x)] * 1.2, qPen+dashed);
draw((0,0,0)..IcoPoints[round(face.y)] * 1.2, qPen+dashed);
draw((0,0,0)..IcoPoints[round(face.z)] * 1.2, qPen+dashed);


draw((0,0,0)..q, qPen+linewidth(0.35mm), EndArrow3);
draw((0,0,0)..n, nPen+linewidth(0.35mm), EndArrow3);

draw(arc(O, q * 0.3, n * 0.3), red, EndArrow3);

label("$n$", n, NW, nPen);
label("$q$", q, W, qPen);
label("$e$", (q + n) / 5, red);

draw(Label("$x$",1),O--X, black, Arrow3);
draw(Label("$y$",1),O--Y, black, Arrow3);
draw(Label("$z$",1),O--Z, black, Arrow3);

