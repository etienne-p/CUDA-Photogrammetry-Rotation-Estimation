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

real pitch = 22.5; // PI / 2.
real yaw = 22.5;
real cumPitch = 0.0;
real cumYaw = 45.0;

srand(24);

int kMax = 3;

triple[] chain = new triple[kMax + 1];
real chainScale = 1.4;
chain[0] = X * chainScale;

pen quadrantPen = royalblue+longdashed+opacity(0.5);

for(int k = 0; k < kMax; ++k)
{
  transform3 basisTransform = rotate(cumYaw, Y) * rotate(cumPitch, X);
  chain[k] = (basisTransform * Z) * chainScale;
  draw_basis(basisTransform);

  transform3 [] transforms = 
  {
    rotate(cumYaw + yaw * -1.0, Y) * rotate(cumPitch + pitch * -1.0, X),
    rotate(cumYaw + yaw * -1.0, Y) * rotate(cumPitch + pitch *  1.0, X),
    rotate(cumYaw + yaw *  1.0, Y) * rotate(cumPitch + pitch *  1.0, X),
    rotate(cumYaw + yaw *  1.0, Y) * rotate(cumPitch + pitch * -1.0, X)
  };

  real zPush = 1.01;
  triple a = transforms[0] * Z * zPush;
  triple b = transforms[1] * Z * zPush;
  triple c = transforms[2] * Z * zPush;
  triple d = transforms[3] * Z * zPush;

  draw(arc(O, a, b), quadrantPen);
  draw(arc(O, b, c), quadrantPen);
  draw(arc(O, c, d), quadrantPen);
  draw(arc(O, d, a), quadrantPen);

  draw(O..a, quadrantPen);
  draw(O..b, quadrantPen);
  draw(O..c, quadrantPen);
  draw(O..d, quadrantPen);

  cumPitch -= pitch;
  cumYaw += yaw * sign(unitrand() - 0.5);

  pitch /= 2.0;
  yaw /= 2.0;
}

draw(O..chain[0], royalblue+dashed);
for(int k = 0; k < kMax - 1; ++k)
{
  draw(O..chain[k+1], royalblue+dashed);
  draw(arc(O, chain[k], chain[k + 1]), royalblue+linewidth(0.3mm), EndArrow3);
}
