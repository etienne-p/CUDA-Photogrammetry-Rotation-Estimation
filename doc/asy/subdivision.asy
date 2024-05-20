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

struct Face
{
  triple A;
  triple B;
  triple C;
}

void draw_face(Face face, pen p) 
{
  draw(arc(O, face.A, face.B), p);
  draw(arc(O, face.B, face.C), p);
  draw(arc(O, face.C, face.A), p);
}

void label_face(Face face, string str, pen p, real coeff)
{
  triple center = unit(face.A + face.B + face.C);
  triple end = center * coeff;
  triple endLabel = end + (0, -0.1, 0.2);
  draw(end--center, p, Arrow3);
  draw(endLabel--end, p);
  label("[" + str + "]", endLabel, E, p);

}

Face get_face_at_index(int index)
{
  triple indices = IcoFaces[index];
  Face face;
  face.A = IcoPoints[round(indices.x)];
  face.B = IcoPoints[round(indices.y)];
  face.C = IcoPoints[round(indices.z)];
  return face;
}

Face get_sub_face(Face face, int index)
{
  Face f;
  if (index == 0)
  {
    f.A = face.A;
  f.B = unit((face.A + face.B) * 0.5);
  f.C = unit((face.C + face.A) * 0.5);
  }
  if (index == 1)
  {
    f.A = unit((face.A + face.B) * 0.5);
  f.B = face.B;
  f.C = unit((face.B + face.C) * 0.5);
  }
  if (index == 2)
  {
    f.A = unit((face.A + face.B) * 0.5);
  f.B = unit((face.B + face.C) * 0.5);
  f.C = unit((face.C + face.A) * 0.5);
  }
  if (index == 3)
  {
    f.A = unit((face.A + face.C) * 0.5);
    f.B = unit((face.B + face.C) * 0.5);
    f.C = face.C;
  }
  return f;
}

pen pHigh = royalblue+linewidth(0.2mm);
pen pLow = mediumgrey;
pen annotation = linewidth(0.2mm)+fontsize(6);

int selectedIndex = 18;

triple face = IcoFaces[selectedIndex];

Face f = get_face_at_index(selectedIndex);


string lbl = string(selectedIndex);

real coeff = 1.2;

pen[] colors = {orange, royalblue, heavygreen, heavymagenta, purple};
int colorIndex = 0;

label_face(f, lbl, annotation+colors[colorIndex], coeff);
draw_face(f, linewidth(0.2mm)+colors[colorIndex]);

int fi = 0;

for(int k = 0; k < 4; ++k)
{
  for(int j = 0; j < 4; ++j)
  {
     Face fs = get_sub_face(f, j);
     pen p = pLow;
     if (j == fi)
     {
       ++colorIndex;
       p = linewidth(0.2mm + 0.04mm * colorIndex)+colors[colorIndex];
       lbl += ":" + string(k);
       coeff += 0.2;
       label_face(fs, lbl, annotation+colors[colorIndex], coeff);
     }
     draw_face(fs, p);
  }
  f = get_sub_face(f, fi);
  ++fi;
  // Skip center face.
  if (fi == 2)
  {
    ++fi;
  }
  fi = fi % 4;
}
