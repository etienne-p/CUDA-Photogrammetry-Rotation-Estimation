size(3inch);
import settings;
settings.render=10;
import three;
import graph3;

pen thickp=linewidth(0.5mm);

currentprojection=orthographic((4,1,2));

draw(unitsphere, lightgrey+opacity(0.1));

draw(unitcircle3, lightgrey+longdashed);
draw(rotate(90, X) * unitcircle3, lightgrey+longdashed);
draw(rotate(90, Y) * unitcircle3, lightgrey+longdashed);

draw(Label("$x$",1),O--X, black, Arrow3);
draw(Label("$y$",1),O--Y, black, Arrow3);
draw(Label("$z$",1),O--Z, black, Arrow3);

real sign(real x)
{
  if (x > 0) return  1.0;
  if (x < 0) return -1.0;
  return 0.0;
}

void draw_basis(transform3 trs) 
{ 
  draw(trs * ((0,0,0)..X), heavyred, EndArrow3);
  draw(trs * ((0,0,0)..Y), heavygreen, EndArrow3);
  draw(trs * ((0,0,0)..Z), heavyblue, EndArrow3);
}

srand(24);

transform3 baseTrs = rotate(12, Z) * rotate(22, X);

real yaw = 45.0;
real cumYaw = 0.0;

int kMax = 3;

triple[] chain = new triple[kMax + 1];
real chainScale = 1.4;
chain[0] = baseTrs * X;


pen quadrantPen = royalblue+longdashed+linewidth(0.05mm);
pen quadrantSurfPen = Cyan+opacity(0.2);

for(int k = 0; k < kMax; ++k)
{
  transform3 trs = baseTrs * rotate(cumYaw + yaw, Z);
  chain[k] = (trs * X) * chainScale;
  draw_basis(trs);

  cumYaw += yaw;

  yaw /= 2.0;
}

draw(baseTrs * unitcircle3, royalblue+dashed);

pen motion = royalblue+linewidth(0.3mm);

draw(baseTrs * shift(Z * 0.75) * scale3(0.2) * unitcircle3, motion, EndArrow3);

draw(O..chain[0], royalblue+dashed);
for(int k = 0; k < kMax - 1; ++k)
{
  draw(O..chain[k+1], royalblue+dashed);
  draw(arc(O, chain[k], chain[k + 1]), motion, EndArrow3);
}




