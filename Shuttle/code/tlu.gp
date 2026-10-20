read("./ideaux_fix.gp");
default(realprecision,120);
n=16;
setrandom(getwalltime());
h=Mod(Pol(vector(n,i,random(17)-8)),'x^n+1);
g=lift(h*polconjL(h,n));
u=logunit(g,n);
print("u=",lift(u));
print("unorm=",polNorme(lift(u),n));
v=lift(u*h);
print("balanced size=",maxPoly(v)," orig=",maxPoly(lift(h)));
quit
