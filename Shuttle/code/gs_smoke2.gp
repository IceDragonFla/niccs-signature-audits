default(realprecision, 120);
read("./ideaux_dbg.gp");
print("read-ok");
n = 64;
t = 8;
facteur = t; facteur2 = 1.09^n;
f = tailleRecursion(n);
print("f=", f);
setrandom(getwalltime());
g = Pol(vector(n, i, random(2*t+1)-t));
nor = (g*polconj(g,n)) % ('x^n+1);
A = matrix(f+4, f);
for(i=1, f+4,
  h = Pol(vector(n, j, random(2*t+1)-t));
  h = (g*h) % ('x^n+1);
  A[i,] = polvec(h, f);
);
print("input ready");
getwalltime();
res = gentry(A, nor, n/f, facteur, facteur2);
print("TIME=", getwalltime(), " ms");
print("res*conj==nor: ", (res*polconjL(res,n)) % ('x^n+1) == Mod(nor,'x^n+1));
print("recovered-lift: ", lift(res));
quit
