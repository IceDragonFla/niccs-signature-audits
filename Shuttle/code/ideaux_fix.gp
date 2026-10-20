fichier="logGS";

\\polconj(U,d) = polcoeff(U,0)-sum(i=1, d-1, polcoeff(U,i)*'x^(d-i));
polconj(U,d)= -Polrev(Vec(U+'x^d))+1+polcoeff(U,0)*(1+'x^d); 
polconjL(U,d) = polconj(lift(U),d);

/*polnorm(U, d) = Mod(U*polconj(U),x^d+1);

polfloor(U) = sum(i=0, poldegree(U), floor(polcoeff(U,i))*x^i);*/

\\tailleRecursion(n)=if(n>1024,16,if(n>64,8,n/*min(4,n)*/)); /* à changer ! */

tailleRecursion(n)=if(n>1024,16,if(n>=256,8,if(n>64,n/16,min(4,n)))); /* à changer ! */

\\automorph(pol,a,T)=sum(i=0,T-1,polcoeff(pol,i)*'x^((i*a)%(2*T)))%('x^T+1);
polAuto(pol,T)=my(a=Vecrev(pol));if(a==0,0,Polrev(vector(#a,i,if(i%2==1,1,-1)*a[i])))
\\diviseP(pol,T)=sum(i=0,T/2,polcoeff(pol,2*i)*'x^i);
diviseP(pol,T,fac,dec)=my(a=Vecrev(pol));if(a==0,0,Polrev(vector((#a-1-dec)\fac+1,i,a[(i-1)*fac+1+dec])))
\\multiP(pol,T,fac)=sum(i=0,T/fac,polcoeff(pol,i)*'x^(i*fac));
multiP(pol,T,fac)=my(a=Vecrev(pol));if(a==0,0,Polrev(vector(#a*fac+1-fac,i,if(i%fac==1,a[(i-1)/fac+1],0))))

vecpol(v,T)=sum(j:small=1,#v,multiP(lift(v[j]),T*#v,#v)*'x^(j-1));
polvec(pol,fac)=vector(fac,i,diviseP(lift(pol),0,fac,i-1));

polNorme(f,N)={
	if(N==1,return(polcoeff(f,0)));
	polNorme(diviseP((f*polAuto(f,N))%('x^N+1),N,2),N/2);
}


fftR(v,n:small,s)={
	my(mu,z,vp,vi,t);
	if(n==0,1/0);
	if(n>1,
		vp=vector(n/2,i,v[2*i-1]);
		vi=vector(n/2,i,v[2*i]);
		vp=fftR(vp,n/2,s);
		vi=fftR(vi,n/2,s);
\\		if(n==4,print("nv=",nv));
		z=exp(-2*I*Pi/n*s);
		mu=1;
		for(i:small=1,n/2,
			t=mu*vi[i];
			v[i+n/2]=vp[i]-t;
			v[i]=vp[i]+t;
			mu*=z;);
			);
	v
};

fftX(V,N,s,pleine)={
	my(v=V);
	if(s==1 && pleine==0,for(i=0,N-1,v[i+1]*=exp(I*Pi*i/N)));
\\	print(v);
	v=fftR(v,N:small,s);
\\	print(nv);
	if(s==-1 && pleine==0,for(i=0,N-1,v[i+1]*=exp(-I*Pi*i/N)));
	v;
}

inv(p,n,pleine)={
\\	if(n==2 && pleine==0,return(Mod(1/polcoeff(p,0),'x^2+1)));
	my(f);
	f=Vecrev(p+'x^(n));
	f=fftX(f,n,1,pleine);
	f=vector(n,i,if(f[i]!=0,1/f[i]/(n)));
	f=fftX(f,n,-1,pleine);
	Mod(real(Polrev(f)),'x^n+if(pleine,-1,1));
}

racine(p,n)={
	my(f);
	f=Vecrev(p+'x^(n));
	f=fftX(f,n,1);
	f=vector(n,i,if(f[i]!=0,sqrt(abs(f[i]))/n));
	f=fftX(f,n,-1);
	Mod(real(Polrev(f)),'x^n+1);
}

/*racineC(p,n)={
	my(f);
	f=Vecrev(p+'x^(n));
	f=fftX(f,n,1);
	f=vector(n,i,if(f[i]!=0,sqrt(f[i])/n));
	f=fftX(f,n,-1);
	Mod(real(Polrev(f)),'x^n+1);
}*/

invracineC(p,n)={
	my(f);
	if(n==1,f=abs(polcoeff(p,0));return(if(f==0,1,1/sqrt(f))));
	f=Vecrev(p+'x^(n));
	f=fftX(f,n,1);
	f=vector(n,i,if(f[i]!=0,1/sqrt(f[i])/n));
	f[2]=conj(f[1]);
	for(i=3,n/2+1,f[i]=conj(f[n+3-i]));
	f=fftX(f,n,-1);
	Mod(real(Polrev(f)),'x^n+1);
}

normalise(p,n)={ /* p/abs(p) */
	my(f);
	f=Vecrev(p+'x^(n));
	f=fftX(f,n,1);
	f=vector(n,i,if(f[i]!=0,f[i]/abs(f[i]),1)/n);
	f=fftX(f,n,-1);
	Mod(real(Polrev(f)),'x^n+1);
}

logunit(g,n)={ /* trouve u tel que g*u^2 petit */
	if(n<=2,return(1));
\\	localprec(30);
	my(u,z,G,G2,logG,c,f,som,ind,j,mu);
	mu=exp(I*Pi/n);
	u=vector(n/2,i,log(mu^lift(Mod(3,2*n)^(i-1))-1));
	z=real(u);
	G=fftX(Vecrev(g+'x^n),n,1);
	for(i=1,n,if(G[i]==0,return(1)));
	G2=vector(n/2,i,j=lift(Mod(3,2*n)^(i-1));G[if(j==1,1,n+1-(j-1)/2)]);
	logG=log(abs(G2));
\\	print(logG," ",z);
	c=lift(Polrev(logG)*inv(Polrev(z),n/2,1));
	som=polcoeff(c%('x-1),0);
\\	print(c);
\\	c=-round((c-som*2/n*(('x^(n/2)-1)/('x-1)))/2);
	c=Vec(c+'x^(n/2));
	for(i=1,n/2,j=c[i]/2-som/n;c[i]=j\1+if(random()/2^31.<j%1,1,0));
	c=-Pol(c);
	c-=c%('x-1);
\\	print(c);
	c=(c*Polrev(u))%('x^(n/2)-1);
\\	print(c);
	c=Vecrev(c+'x^(n/2));
	f=vector(n,i,0);
	for(i=1,n/2,j=lift(Mod(3,2*n)^(i-1));ind=if(j==1,1,n+1-(j-1)/2);f[ind]=exp(c[i]);ind=n+1-(2*n-j-1)/2;f[ind]=conj(exp(c[i])));
\\	print(f);
	c=fftX(f,n,-1);
\\	print(c/n);
	Polrev(round(real(c/n)));
}

maxPoly(p)=vecmax(abs(Vec(if(p==0,2,p))));

petitEuclide(f,g,N)={my(U,V,e,q,pr:real=30,quo);
\\	print("petit:",f," ",g);
	U=[1,0];
	V=[0,1];
	while(g!=0,
\\		print([f,g]," ",lift(f*inv(g,N)), " ",U," ",V);
		e=1;
		pr+=0.01;
		while(e>-1,localprec(truncate(pr):small);quo=lift(f*inv(g,N));q=round(quo,&e);for(i=0,N-1,if(polcoeff(quo,i)==polcoeff(quo,i)+1e-2,e=0));if(e>-1,pr*=2););
		[U,V]=[V,U-q*V%('x^N+1)];
		[f,g]=[g,f-q*g%('x^N+1)];
		);
	e=1;
	while(e>-1,localprec(truncate(pr):small);q=round(lift(inv(f,N)),&e);if(e>-1,pr*=2););
\\	print("petit fin:",f," ",q," ",U);
	if(polNorme(f,N)==1,[U[1]*q%('x^N+1),U[2]*q%('x^N+1),1],[0,0,2]);
}

euclideEtendu(f,g,N)={my(Nf,Ng,U,V,u,v,d,maxi,k);
	if(N==1, gcdext(polcoeff(f,0),polcoeff(g,0)),
	Nf=(f*polAuto(f,N))%('x^N+1);
	Ng=(g*polAuto(g,N))%('x^N+1);
	[u,v,d]=euclideEtendu(diviseP(Nf,N,2),diviseP(Ng,N,2),N/2);
	if(d!=1,if(N<=4 && maxPoly(f)<2.^3000,return(petitEuclide(f,g,N)),return([0,0,2])));
\\	print([Nf,Ng]," ",[u,v]);
	[u,v]=[multiP(u,N,2),multiP(v,N,2)];
	U=(u*polAuto(f,N))%('x^N+1);
	V=(v*polAuto(g,N))%('x^N+1);
	\\default(format,"g.9");
	maxi=log(2+max(maxPoly(U),maxPoly(V)))/log(2);
	localbitprec(truncate(maxi*3)+20);
	k=round((V*polconj(f,N)-U*polconj(g,N))*inv((f*polconj(f,N)+g*polconj(g,N)+0.)%('x^N+1),N),&u);
	[U,V]=lift([U+k*g,V-k*f]);
\\	print([f,g]," ",[U,V]);
	[U,V,1];
)}

petitlll(L,n)={my(U,m,nouv,d0,ancd0,u,mu,e,m2,diff,iter,Lanc,Udeb,Ldeb,invd0);
\\	print(L);
	U=[Mod(1,'x^n+1),0;0,Mod(1,'x^n+1)];
	L*=invracineC(matdet(lift(L))%('x^n+1),n);
	Lanc=L;
\\	print("début ",lift(L));
	ancd0=-1;
	mu=polconjL(L[1,1],n)*L[1,2]+polconjL(L[2,1],n)*L[2,2];
	while(1,
\\	print(U);
		Ldeb=L;
		Udeb=U;
		d0=polconjL(L[1,1],n)*L[1,1]+polconjL(L[2,1],n)*L[2,1];
		invd0=inv(lift(d0),n);
		m=lift(mu*invd0);
		m2=round(m,&e);
		diff=m-m2;
		iter=0;
		while(abs(polNorme(lift(m-m2),n))>0.998,
			u=logunit(diff,n)*prod(i=1,0,e=random(n/2);('x^(2*e+1)-1)/('x-1))%('x^n+1);
			m2=round(m,&e)+round((diff*u)%('x^n+1),&e)*round(inv(lift(u),n),&e);
			iter++;
			if(iter==100,print("échec");return(U));
			);
		mu-=m2*d0;
\\		if(polcoeff(lift(mu2),0)==0,return(U));
		nouv=abs(polNorme(lift(invd0*(1+mu*polconjL(mu,n))),n));
\\		print(lift(mu*invd0)," d0=",lift(d0)," ancd0=",lift(ancd0)," nouv=",nouv," ",0);
\\		breakpoint;
		if(nouv>0.999*ancd0 && ancd0>0,
\\			breakpoint;
\\			break;
			if(polNorme(lift(L[1,1]),n)<10.^n*max(1,polNorme(lift(L[2,1]),n)), /*print("2 L=",L);*/ break,
				u=logunit(lift(d0),n);
				U[,1]*=u;
				u=round(lift(inv(u,n)));
				U[,2]*=u;
				L=Lanc*U;
				mu=polconjL(L[1,1],n)*L[1,2]+polconjL(L[2,1],n)*L[2,2];
				ancd0=polNorme(lift(d0),n);
				m=mu*invd0;
				if(maxPoly(u)<100,break);
				next;
			);
		);
		round(diff,&e);
		if(nouv==0 || e>-1 || (n>4 && vecmax(matrix(2,2,i,j,maxPoly(lift(U[i,j]))))>10.^(precision(1.)/1.)),
\\			print("e=",e," L=",lift(L));
\\			print(lift(Lanc*U));
			return(U)); /* précision */
		ancd0=nouv;
/*		print(lift(mu*polconjL(mu,n)));*/

		mu=polconjL(mu,n);
		nouv=U[,2]-m2*U[,1];
		U[,2]=U[,1];
		U[,1]=nouv;
		nouv=L[,2]-m2*L[,1];
		L[,2]=L[,1];
		L[,1]=nouv;
		L=Lanc*U;
		mu=polconjL(L[1,1],n)*L[1,2]+polconjL(L[2,1],n)*L[2,2];
\\		f=ancd0/polNorme(lift(polconjL(L[1,1],n)*L[1,1]+polconjL(L[2,1],n)*L[2,1]),n);
\\		if(f<1e-3 || f>1e3,breakpoint());
	);	
\\	print("FIIIIIIIIIN ",N," ",lift(L));
\\	print("3 L=",lift(L));
\\	print(lift(Lanc*U));
\\	breakpoint;
\\print(U);
	U
}

petitlllrapide(L,n)={my(U,Ut);
	if(precision(1.)<100,
		petitlll(L,n),
\\		print(precision(1.)," ",L);
		if(1,localprec(precision(1.)\2);U=petitlllrapide(L*1.,n););
\\		Lanc=L;
		L=L*U;
		localprec(min(precision(1.)\2,precision(1.)-truncate(log(vecmax(matrix(2,2,i,j,maxPoly(lift(U[i,j])))))/log(10))));
		Mod(U*petitlllrapide(L*1.,n),'x^n+1);
		);
}

etend(A,n,f)={
	my(P,k2:small,l2:small);
	matrix(#A*f,#A*f,k,l,k2=k-1;l2=l-1;P=A[1+k2\f,1+l2\f];
	sum(z:small=0,n/f-1,if(k2%f>l2%f && z==0,-1,1)*polcoeff(P,(z*f+n+l2%f-k2%f)%n)*'x^z));
}

moyenlll(G,n:small,niveau:small)={
	my(U,Ut,L,Orth,invLi,nouvnorm,u,delta,maxi,mu,Gn,Ginit);
	my(m,i,j,nbRecule=0,nbReculeAnc,nbEchMoyen=0,nbEchMoyenAnc,change,f):small;
	m=length(G);
	Ut=matrix(m, m, i, j, i==j);
	Ginit=G;

	while(n>1,
		if(n==16 && m>40,localprec(20),localprec(10));
		U=matrix(m, m, i, j, i==j);
		L=matrix(m, m, i, j, Mod(i==j,'x^n+1)*1.);
		Orth=matrix(m, m, i, j, Mod(i==j,'x^n+1)*1.);
		invLi=vector(m,i,0);
		i=1;
		delta=1.+.2/n;
\\		print(n," ",m," ",delta);
		nbEchMoyenAnc=nbEchMoyen;
		nbReculeAnc=nbRecule;
		while(i<=m && nbRecule-nbReculeAnc<=(nbEchMoyen-nbEchMoyenAnc)/2.+5,
			L[i,]=Mod(G[i,],'x^n+1)*1.;
			for(j:small=1,i-1,u=sum(z:small=j,m,polconjL(Orth[j,z],n)*L[i,z]);for(z:small=j,m,L[i,z]-=Orth[j,z]*u););
			
			nouvnorm=sum(z:small=i,m,polconjL(L[i,z],n)*L[i,z]); /* carrée ! */
			u=logunit(lift(nouvnorm),n);
			for(k:small=1,m,U[i,k]=(U[i,k]*u)%('x^n+1));
			for(k:small=1,m,L[i,k]*=u);
			for(k:small=1,m,G[i,k]=(G[i,k]*u)%('x^n+1));
			nouvnorm*=u*polconjL(u,n);
			maxi=1;
			forstep(j:small=i-1,1,-1,
				mu=round(lift(L[i,j]*(invLi[j])),&u);
				maxi=max(maxi,maxPoly(mu));
	\\			print(mu," ",L[i,1]);
				for(k:small=1,j,L[i,k]-=mu*L[j,k]);
				for(k:small=1,m,U[i,k]-=(mu*U[j,k])%('x^n+1));
				for(k:small=1,m,G[i,k]-=(mu*G[j,k])%('x^n+1));
			);
			if(maxi>1e10,/*print("rest");*/next);
	\\		print(lift(L[i,1..i-1])," ",lift(nouvnorm));
	\\		print(vector(i-1,k,abs(polNorme(lift(L[k,k]),n)^(1/n)))," ",abs(polNorme(lift(nouvnorm),n)^(1/n/2)));
			j=i;
			change=0;
			while(j>1 && polNorme(lift(polconjL(L[j-1,j-1],n)),n)^2>delta*polNorme(lift(polconjL(L[i,j-1],n)*L[i,j-1]+nouvnorm),n),
				nbEchMoyen++;
				for(k:small=1,m,[U[j-1,k],U[j,k]]=[U[j,k],U[j-1,k]]);
				for(k:small=1,m,[G[j-1,k],G[j,k]]=[G[j,k],G[j-1,k]]);
				nouvnorm+=polconjL(L[i,j-1],n)*L[i,j-1];
				j--;
				u=sum(z:small=j,m,polconjL(Orth[j,z],n)*L[i,z]);for(z:small=j,m,L[i,z]-=Orth[j,z]*u);\\print(lift(L[i,1..m]));
				change=1;
			);
			if(j>1 && change==0 && polNorme(lift(L[j-1,j-1]),n)^2>10000^n*polNorme(lift(nouvnorm),n),
	\\			print(polNorme(lift(L[i,j-1]*invLi[j-1]),n));
\\				print("brouf ",i," ",vector(m,k,abs(polNorme(lift(L[k,k]),n)^(1/n))));
				i=j-1;
				nbRecule++;
				next);
			if(change,i=j;next);
			if(0,
				L[i,]=Mod(G[j,],'x^n+1)*1.;
				for(j2:small=1,j-1,u=sum(z:small=j2,m,polconjL(Orth[j2,z],n)*L[i,z]);for(z:small=j2,m,L[i,z]-=Orth[j2,z]*u););
				);
	\\		print(lift(tmpL-L[i,1..m]));
	\\		print("");
	\\		L[i,]=tmpL;
			for(k:small=1,j-1,L[j,k]=L[i,k]);
			for(z:small=j,m,Orth[j,z]=L[i,z]);
			nouvnorm=racine(lift(nouvnorm),n);
			nouvnorm*=-normalise(lift(L[i,j]),n);
			L[j,j]=nouvnorm;
			invLi[j]=inv(lift(nouvnorm),n);
	\\		print("v2 L=",lift(L[j,]));
			Orth[j,j]-=nouvnorm;
			u=inv(lift(racine(lift(sum(z:small=j,m,polconjL(Orth[j,z],n)*Orth[j,z])/2.),n)),n);
			for(z:small=j,m,Orth[j,z]*=u);
			i++;
\\			if(i==m+1 && delta>1.01,print("RECOM");delta=1.005;i=1);
		);
		u=vector(m,k,abs(polNorme(lift(L[k,k]),n)^(1/n)));
\\		print("FIN ",u);
		f=if(n>8,2,n);
		Gn=etend(round(G*1.3^(n*m)/vecmin(u),&u),n,f);
		Ut=(U*Ut)%('x^n+1);
		U=etend(Ut,n,f);
		Ut=matrix(m*f,m*f);
		for(i=0,1,for(j=1,m/2,for(k=1,f,Ut[,(k-1+i*f)*m/2+j]=U[,(j-1+i*m/2)*f+k];)));
		G=Gn;
		n\=f;
		m*=f;
	);
	return(Ut);
	print("fin");
	Ut=qflll(matrix(m,m,i,j,polcoeff(G[j,i],0)))~*Ut;
	Ut;
}

fastlll(G,n:small,deter:real,niveau:small=1)={
	my(U,L,Gn,V,A,B,C,D,mu,Gm,g,u,maxi,mini,bit,invLi,vec,nouvnorm,det,logGS,Orth,maxTrans,invRT,muM,Ginit,curbit,r);
	my(m,f,iter,ident,fini,z,borne,nbTours,maxGratte=1,dernGS):small;
	m=length(G);
	U=matrix(m, m, i, j, i==j);
	L=matrix(m, m, i, j, Mod(i==j,'x^n+1)*1.);
	Orth=matrix(m, m, i, j, Mod(i==j,'x^n+1)*1.);
	invRT=L;
	invLi=vector(m,i,0);
	curbit=vector(2,i,bitprecision(1.));
	nbTours=if(niveau==1,2,if(n<32,truncate(m^2),truncate(m^2)));
\\	M=matrix(m,m,i,j,if((i-m/2-.1)*(j-m/2-.1)<=0 && (i+j)%(m/2)==1,if(i<=m/2,1,-1),0));

	if(niveau==2,nbTours*=1);
	if(m*n>=128,maxGratte=2);
	fini=0;
	maxGratte=1;
	for(tour:small=1,nbTours,
		Ginit=G;
		localprec(truncate(curbit[1+tour%2]*log(2)/log(10)));
		if(niveau<=2,print("tour n=",n," m=",m," ",tour," ",bitprecision(1.)," maxTrans=",maxTrans," ",1.*maxTrans/vecmin(curbit)));
\\		for(i=1,m,if(G[i,i]!=polconj(G[i,i],n),print("A ",i);1/0));
		gettime();
\\		if(n>4,print(G-round(G));if(tour>500,1/0));
\\		print(U);
\\		print(matrix(m,m,i,j,log(1+maxPoly(lift(G[i,j])))/log(10))~);
		if(tour==1 || maxTrans>vecmin(curbit)*0.1 || tour>=dernGS+maxGratte,
\\			if(tour>dernGS,logGS=vector(m/2+1,i,abs(polNorme(lift(L[i,i]),n)^(1/n)));print(logGS));
			dernGS=tour;
			maxTrans=0;
			muM=matrix(m,m/2,i,j,i==j);
\\			ancU=U;
\\			print(U);
			for(i:small=1,m,
				L[i,]=Mod(G[i,],'x^n+1)*1.;
				for(j:small=1,min(i-1,min(m,m+maxGratte-i)),u=sum(z:small=j,m,polconjL(Orth[j,z],n)*L[i,z]);for(z:small=j,m,L[i,z]-=Orth[j,z]*u););
				
				nouvnorm=sum(z:small=i,m,polconjL(L[i,z],n)*L[i,z]); /* carrée ! */
\\				if(i==m/2+1,print(racine(lift(nouvnorm),n)));
	\\			print(nouvnorm);
	\\			print("L=",lift(L[i,]));
	\\			print(i," ",lift(G[i,]));
	\\			print(i," ",lift(L[i,]));
	/*			print("L=",lift(L));
				print("G=",G);*/
\\		print(i," ",j," ",U~*M*U%('x^n+1));
				if(i<=m/2,
					u=logunit(lift(nouvnorm),n);
\\					print("u=",u);
	\\				u=1;
					for(k:small=1,m,U[i,k]=(U[i,k]*u)%('x^n+1));
					for(k:small=1,i,L[i,k]*=u);
					for(k:small=1,m,G[i,k]=(G[i,k]*u)%('x^n+1));

					u=round(lift(inv(u,n)));
					for(k:small=1,m,U[m+1-i,k]=(U[m+1-i,k]*u)%('x^n+1));
					for(k:small=1,m,G[m+1-i,k]=(G[m+1-i,k]*u)%('x^n+1));
\\		print(i," ",j," ",U~*M*U%('x^n+1));
					forstep(j:small=i-1,1,-1,
						mu=round(lift(L[i,j]*(invLi[j])),&u);
						muM[i,j]=mu;
						for(k:small=1,j,L[i,k]-=mu*L[j,k]);
						for(k:small=1,m,U[i,k]-=(mu*U[j,k])%('x^n+1));
						for(k:small=1,m,G[i,k]-=(mu*G[j,k])%('x^n+1));
\\						for(k:small=1,m,U[m+1-j,k]+=(mu*U[m+1-i,k])%('x^n+1));

					)
					,
					my(comb);
					comb=vector(m/2);
					forstep(j:small=i-1,m/2+1,-1,
						comb[j-m/2]=mu=muM[m+1-j,m+1-i]-sum(k:small=j+1,i-1,comb[k-m/2]*muM[m+1-j,m+1-k])%('x^n+1);
	\\					print(mu);
						for(k:small=1,m,U[i,k]+=(mu*U[j,k])%('x^n+1)); \\ réduction en taille U^-s
						for(k:small=1,m,G[i,k]+=(mu*G[j,k])%('x^n+1));
						for(k:small=1,m+1-i,L[i,k]+=mu*L[j,k]);
						for(k:small=1,m+1-i,muM[i,k]+=(mu*muM[j,k])%('x^n+1));
						);
					forstep(j:small=i-1,m/2+1,-1,
						L[i,m+1-j]=0;
						mu=muM[j,m+1-i]-sum(k:small=m/2+1,i-1,comb[k-m/2]*muM[k,m+1-j])%('x^n+1); \\ réduction en taille coin symétrique; 
						muM[i,m+1-j]=muM[j,m+1-i];
\\						print("coin ",mu);
						for(k:small=1,m,U[i,k]-=(mu*U[m+1-j,k])%('x^n+1));
						for(k:small=1,m,G[i,k]-=(mu*G[m+1-j,k])%('x^n+1));
						for(k:small=1,m+1-i,L[i,k]-=mu*L[m+1-j,k]);
						);
\\					print(comb);
\\					print(lift(L[i,1..m+1-i]));
		\\			print(matrix(m,m,i,j,log(1+maxPoly(lift(G[i,j])))/log(10))~);
		\\			print(matrix(m/2,m/2,i,j,log(1+maxPoly(lift(L[i+m/2,j]*invLi[j])))/log(10))~);
					forstep(j:small=m+1-i,1,-1,invRT[i,j]=(L[i,j]-sum(k:small=j+1,m+1-i,invRT[i,k]*L[k,j]))*invLi[j];);
					for(j:small=1,m+1-i,
						mu=round(lift(invRT[i,j]),&u);
\\						print("nouv ",mu,lift(L[j,1..j]));
\\						if(0,mu=0);
						maxTrans=maxTrans || u>1;
						for(k:small=1,j,L[i,k]-=mu*L[j,k]);
\\						print(lift(L[i,1..m+1-i]));
						for(k:small=1,m,U[i,k]-=(mu*U[j,k])%('x^n+1));
						for(k:small=1,m,G[i,k]-=(mu*G[j,k])%('x^n+1));
						muM[i,j]=muM[i,j]+mu;
\\						print("nouv ",mu);
						);
					if(maxTrans && niveau==2,maxTrans=0;print("recalc ",i),next;);
					);
	\\			print("cur L=",lift(L[i,1..i-1]));

				L[i,]=Mod(G[i,],'x^n+1)*1.;
				for(j:small=1,i-1,u=sum(z:small=j,m,polconjL(Orth[j,z],n)*L[i,z]);for(z:small=j,m,L[i,z]-=Orth[j,z]*u););
				if(i>m/2,next);
	\\			Orth[m,]=L[i,];
\\				print("cur L=",lift(L[i,1..i-1]));
\\				print("fin ",lift(L[i,i..m]));
				for(z:small=i,m,Orth[i,z]=L[i,z]);
				nouvnorm=racine(lift(sum(z:small=i,m,polconjL(L[i,z],n)*L[i,z])),n);
\\				if(i==m/2+1,print(nouvnorm));
				nouvnorm*=-normalise(lift(L[i,i]),n);
				L[i,i]=nouvnorm;
				invLi[i]=inv(lift(nouvnorm),n);
	\\			print("v2 L=",L[i,]);
				Orth[i,i]-=nouvnorm;
				u=inv(lift(racine(lift(sum(z:small=i,m,polconjL(Orth[i,z],n)*Orth[i,z])/2.),n)),n);
	\\			print("u=",u);
				for(z:small=i,m,Orth[i,z]*=u);
	\\			print("O=",Orth[i,i..m]);
	\\			u=sum(z:small=i,m,polconjL(Orth[i,z],n)*Orth[m,z]);for(z:small=i,m,Orth[m,z]-=Orth[i,z]*u);
	\\			print("image=",lift(nouvnorm)," vs. ",lift(Orth[m,i..m]));
			);
\\			print(lift(L));
\\			print(L[m/2+1,m/2]);
\\			breakpoint;
\\			print(matrix(m,m,i,j,log(1+maxPoly(lift(G[i,j])))/log(10))~);
\\			print(L[m/2+1,m/2]);
\\			print(lift(L));
			if(maxTrans,tour--;next);

			for(i:small=m/2+1,min(m,m/2+maxGratte),
				L[i,i]=deter*invLi[m+1-i];
				forstep(j:small=i-1,m+1,-1,
					L[i,j]=-deter*invLi[m+1-i];
					);
				);

\\			print("L=",lift(L));
	\\		print("G=",G);
			maxi=1+vecmax(matrix(m,m,i,j,maxPoly(G[i,j])));
			mini=vecmin(vector(m/2+1,i,abs(polNorme(lift(L[i,i]),n))))^(1/n);
\\			if(log(maxi/mini)<n^2 && tour%2==0,break);
			curbit[1+tour%2]=truncate(log(maxi/mini)/log(2)*1.3+5*sqrt(n)+1*m*log(n+1)/log(2));
			if(niveau==2,print(truncate(log(maxi/mini)/log(2)*1.3)));
			localprec(truncate(curbit[1+tour%2]*log(2)/log(10)));
	\\		if(niveau==2,print(curbit," ",log(maxi)/log(2));print(matrix(m,m,i,j,log(maxPoly(G[i,j]))/log(2))));
	\\		for(i=1,m,if(G[i,i]!=polconj(G[i,i],n),print("B ",i);1/0));
			,
			L[m/2+1,m/2+1]=deter*invLi[m/2];
		);
		if(niveau<=2,
			logGS=vector(m/2+1,i,abs(polNorme(lift(L[i,i]),n)^(1/n)));
\\			if(m<8,breakpoint);
			if(niveau==2,for(i=1,m/2+1,write(fichier,log(logGS[i]/sqrt(deter))/log(2)+1e-3));write(fichier,"\n"));
			print(logGS));
		ident=1;
\\		print(matrix(m,m/2,i,j,if(i>j,log(1+maxPoly(lift(L[i,j]*invLi[j])))/log(2),0))~);
		if(niveau==1,
			Gm=lift([L[1,1],0;L[2,1],L[2,2]]);
			Gm=lift(Gm*invracineC((Gm[1,1]*Gm[2,2])%('x^n+1),n));
			print("niveau 1");
			maxi=1+max(max(maxPoly(Gm[1,1]),maxPoly(Gm[2,1])),1);
			mini=min(abs(polNorme(Gm[1,1],n)),abs(polNorme(Gm[2,2],n)))^(1/n);
			bit=-2*log(mini)/log(2);
			f=tailleRecursion(n):small;
			my(mult,ndeter):real;
			mult=2^(40+6*sqrt(n))/mini;
			ndeter=mult^2;
			mult=min(mult/sqrt(deter),1);Gm=G;ndeter=mult^2*deter;
			Gn=etend(Gm,n,f);
\\				print("récursion");
			print(mult^f);
			Gn=round(Gn*mult,&u);
\\				localprec(truncate((bit+40)*log(2)/log(10)));
\\				print(n," ",f," ",fplll," ",Mod(Gm[1,1],'x^n+1)*mult^2);
			V=fastlll(Gn,n/f,ndeter,niveau+1);
			gettime();
			localprec(truncate(curbit[1+tour%2]*log(2)/log(10)-1));
			g=2;
			z=1;
			borne=1;
			iter=0;
			while(g!=1,
				iter++;
				if(iter>2,borne++;iter=0;if(borne>=3,borne=1;z++);/*print(z," ",V)*/);
				vec=vector(2*f,y,if(y<=z,random(2*borne+1)-borne,0))*V;
				A=sum(z:small=0,f-1,multiP(lift(vec[1+z]),n,f)*'x^z);
				B=sum(z:small=0,f-1,multiP(lift(vec[f+1+z]),n,f)*'x^z);
				[C,D,g]=euclideEtendu(A,B,n);
\\					if(n==64,print("j=",j," A=",A," ",B," ",g));
\\					if(g>1 && iter>90 && z==1,print(g," ",vec);if(iter==99,1/0););
				);
			Gm=lift([L[1,1],0;L[2,1],L[2,2]]);
			Gn=[A,B]*Gm%('x^n+1);
			nouvnorm=sqrt(polNorme(lift(polconjL(Gn[1],n)*Gn[1]+polconjL(Gn[2],n)*Gn[2])%('x^n+1),n));
\\				print(Gn," ",nouvnorm);
			if(nouvnorm>polNorme(lift(L[1,1]),n),A=C=1;B=D=0;nouvnorm=polNorme(lift(L[1,1]),n));
			if(nouvnorm^2>polNorme(lift(L[2,2]*polconjL(L[2,2],n)+L[2,1]*polconjL(L[2,1],n)),n),A=C=0;B=D=1);
			[C,D]=[-D,C];
\\				print("fin: ",[A,B]);
\\				print([C,D]);
			[A,B,C,D]=Mod([A,B,C,D],'x^n+1);
			V=[A,C;B,D];
\\				print("V=",lift(V));
			for(z:small=1,m,[U[1,z],U[2,z]]=lift([U[1,z],U[2,z]]*V););
			for(z:small=1,m,[G[1,z],G[2,z]]=lift([G[1,z],G[2,z]]*V););
			,
			parfor(num:small=0,if(m==2,0,m\4-1),
				localprec(truncate(curbit[1+tour%2]*log(2)/log(10)-1));
				my(ind,V);
				ind=if(m==2,1,1+tour%2)+2*num;
				mu=round(lift(L[ind+1,ind]*invLi[ind]),&u);
				L[ind+1,ind]-=mu*L[ind,ind];
				Gm=lift([L[ind,ind],0;L[ind+1,ind],L[ind+1,ind+1]]);
				if(n>16,
					my(fplll:small=0);
					Gm=lift(Gm*invracineC((Gm[1,1]*Gm[2,2])%('x^n+1),n));
					maxi=1+max(max(maxPoly(Gm[1,1]),maxPoly(Gm[2,1])),1);
					mini=min(abs(polNorme(Gm[1,1],n)),abs(polNorme(Gm[2,2],n)))^(1/n);
					bit=-2*log(mini)/log(2);
					f=tailleRecursion(n):small;
					if(bit*10<n*7 && n<=64 && bitprecision(1.)<=(n*m*2)\3,f=max(1,n/16);fplll=1;print(ind,": ",maxi," ",mini));

	\\				print(log(maxi)/log(2)," ",log(mini)/log(2)," ",L[i+1,i+1]);
	\\				if(n>30,print(Gm);print(lift(L[i+1,i+1])));
					my(mult,ndeter):real;
					mult=2^(40+6*sqrt(n))/mini;
					ndeter=mult^2;
					if(niveau==1,mult=min(mult/sqrt(deter),1);Gm=G;ndeter=mult^2*deter);
					Gn=etend(Gm,n,f);
	\\				print("récursion");
					if(niveau==1 && mult<1,print(mult^f));
					Gn=round(Gn*mult,&u);
	\\				localprec(truncate((bit+40)*log(2)/log(10)));
	\\				print(n," ",f," ",fplll," ",Mod(Gm[1,1],'x^n+1)*mult^2);
					if(!fplll,V=fastlll(Gn,n/f,ndeter,niveau+1),
						V=moyenlll(Gn,n/f,niveau+1);
						f=n;
						);
					gettime();
					localprec(truncate(curbit[1+tour%2]*log(2)/log(10)-1));
					g=2;
					z=1;
					borne=1;
					iter=0;
					while(g!=1,
						iter++;
						if(iter>2,borne++;iter=0;if(borne>=3,borne=1;z++);/*print(z," ",V)*/);
						vec=vector(2*f,y,if(y<=z,random(2*borne+1)-borne,0))*V;
						A=sum(z:small=0,f-1,multiP(lift(vec[1+z]),n,f)*'x^z);
						B=sum(z:small=0,f-1,multiP(lift(vec[f+1+z]),n,f)*'x^z);
						[C,D,g]=euclideEtendu(A,B,n);
	\\					if(n==64,print("j=",j," A=",A," ",B," ",g));
	\\					if(g>1 && iter>90 && z==1,print(g," ",vec);if(iter==99,1/0););
						);
					Gm=lift([L[ind,ind],0;L[ind+1,ind],L[ind+1,ind+1]]);
					Gn=[A,B]*Gm%('x^n+1);
					nouvnorm=sqrt(polNorme(lift(polconjL(Gn[1],n)*Gn[1]+polconjL(Gn[2],n)*Gn[2])%('x^n+1),n));
	\\				print(Gn," ",nouvnorm);
					if(nouvnorm>polNorme(lift(L[ind,ind]),n),A=C=1;B=D=0;nouvnorm=polNorme(lift(L[ind,ind]),n));
					if(nouvnorm^2>polNorme(lift(L[ind+1,ind+1]*polconjL(L[ind+1,ind+1],n)+L[ind+1,ind]*polconjL(L[ind+1,ind],n)),n),A=C=0;B=D=1);
					[C,D]=[-D,C];
	\\				print("fin: ",[A,B]);
	\\				print([C,D]);
					[A,B,C,D]=Mod([A,B,C,D],'x^n+1);
					V=[A,C;B,D];
	\\				print("V=",lift(V));
				,
					V=petitlllrapide([L[ind,ind],L[ind+1,ind];0,L[ind+1,ind+1]],n);
					V=Mod(V,'x^n+1);
					det=lift(V[1,1]*V[2,2]-V[1,2]*V[2,1]);
					V[,1]*=det;
	\\				print("L=",lift([L[i,i],L[i+1,i];0,L[i+1,i+1]]));
	\\				print(lift(V));
					);
				V
				,
				V,
				my(i=if(m==2,1,1+tour%2)+2*num);
\\				print(i," ",V);
				Gm=lift([L[i,i],0;L[i+1,i],L[i+1,i+1]]);
				mu=round(lift(L[i+1,i]*invLi[i]),&u);
				det=lift(V[1,1]*V[2,2]-V[1,2]*V[2,1]);
				if(polNorme(det,n)!=1,print("ERRRRREEUR: ",n," ",det," ",V);1/0);
				if(V[2,1]!=0,ident=0;/*print(i," ",lift(V))*/);
				Gm=Gm~*V;
				r=racine(lift(polconjL(Gm[1,1],n)*Gm[1,1]+polconjL(Gm[2,1],n)*Gm[2,1]),n);
				L[i,i]=r;
	\\			if(niveau==2,print(i," ",polNorme(lift(r),n)^(1/n)));
				r=inv(lift(r),n);
				L[i+1,i]=Gm[1,2];L[i+1,i+1]=Gm[2,2];
				Gm=[0,0;-Gm[2,1]*r,Gm[1,1]*r];
				Gm[1,]=[polconjL(Gm[2,2],n),-polconjL(Gm[2,1],n)];
				Gm=Gm~;
				for(z:small=i+1,min(m,i+maxGratte+dernGS-tour),[L[z,i],L[z,i+1]]=[L[z,i],L[z,i+1]]*Gm);
	\\			print(lift(L[i+1,i+1])," ",lift(L[i+1,i]));
				u=round(lift(L[i+1,i]*r),&det);
				u=0;
				if(det>1,print("précision"););
				L[i+1,i]-=u*L[i,i];
				V[,2]-=V[,1]*u;
				V[1,]-=V[2,]*mu;
				invLi[i]=r;
				invLi[i+1]=inv(lift(L[i+1,i+1]),n);
	\\			print(lift(V));
	\\			print(u," ",mu);
				maxTrans=max(maxTrans,log(vecmax(matrix(2,2,i,j,maxPoly(lift(V[i,j])))))/log(2));
	/*			print("V=",lift(V)," G=",[G[i,i],G[i+1,i];polconj(G[i+1,i],n),G[i+1,i+1]]);*/
	\\			print(L[i,i]," ",L[i+1,i+1]);
				for(z:small=max(1,i-maxGratte+tour-dernGS+1),i-1,[L[i,z],L[i+1,z]]=[L[i,z],L[i+1,z]]*V;);
				for(z:small=1,m,[U[i,z],U[i+1,z]]=lift([U[i,z],U[i+1,z]]*V););
				for(z:small=1,m,[G[i,z],G[i+1,z]]=lift([G[i,z],G[i+1,z]]*V););
				if(i<m/2,
	\\				print(i," ",V);
					V=[V[1,1],-V[1,2];-V[2,1],V[2,2]];
					for(z:small=max(1,m-i+1-maxGratte+tour-dernGS),m-i+1,[L[m-i,z],L[m-i+1,z]]=[L[m-i,z],L[m-i+1,z]]*V;);
					Gm=[Gm[1,1],-Gm[1,2];-Gm[2,1],Gm[2,2]];
					for(z:small=m-i,min(m,m-i+maxGratte+dernGS-tour),[L[z,m-i],L[z,m-i+1]]=[L[z,m-i],L[z,m-i+1]]*Gm);
					for(z:small=1,m,[U[m-i,z],U[m-i+1,z]]=lift([U[m-i,z],U[m-i+1,z]]*V););
					for(z:small=1,m,[G[m-i,z],G[m-i+1,z]]=lift([G[m-i,z],G[m-i+1,z]]*V););
				);
			);
			maxTrans=max(maxTrans,log(vecmax(matrix(m,m/2,i,j,if(j<i,maxPoly(lift(L[i,j]*invLi[j])),1))))/log(2));
	\\		print(matrix(m,m/2,i,j,if(i>j,log(1+maxPoly(lift(L[i,j]*invLi[j])))/log(2),0))~);
			if(ident,fini++;if(fini>3,break),fini=0);
	\\		print("G=",G);
		);
	);
	U;
}

generecombi(A,B,n:small)={
	my(a,b);
	a=(vector(#A~,i,random(2*'x^n))*A)%('x^n+1);
	b=(vector(#B~,i,random(2*'x^n))*B)%('x^n+1);
	polvec((vecpol(a,n)*vecpol(b,n))%('x^(n*#a)+1),#a);
}

ideauxmul(I1,I2,norme,n:small,facteur:int)={
	my(G,petitG,L,Gn,V,A,B,C,D,mu,Gm,g,u,maxi,mini,bit,invLi,vec,nouvnorm,det,logGS,Orth,maxTrans,invRT,invRnorme,Ginit,curbit,r);
	my(m,M,f,iter,ident,fini,z,borne,nbTours,maxGratte=1,dernGS,stock):small;
	m=length(I1);
	stock=3;
	M=m+1;
	G=matrix(M,m);
	petitG=matrix(stock,m);
	L=matrix(M, M, i, j, Mod(i==j,'x^n+1)*1.);
	invLi=vector(M,i,0);
	Orth=matrix(M, m, i, j, Mod(i==j,'x^n+1)*1.);
	curbit=vector(2,i,bitprecision(1.));
	invRnorme=invracineC(lift(norme),n*m);
	for(i=0,m-1,G[1+i,]=polvec((vecpol(I1[1+i%#I1~,],n)*vecpol(I2[1+i%#I2~,],n))%('x^(n*m)+1),m));
\\	M=matrix(m,m,i,j,if((i-m/2-.1)*(j-m/2-.1)<=0 && (i+j)%(m/2)==1,if(i<=m/2,1,-1),0));

	for(tour:small=1,oo,
		Ginit=G;
		localprec(truncate(curbit[1+tour%2]*log(2)/log(10)));
		print("tour n=",n," m=",m," ",tour," ",bitprecision(1.));
\\		for(i=1,m,if(G[i,i]!=polconj(G[i,i],n),print("A ",i);1/0));
		gettime();
		G[m+1,]=generecombi(I1,I2,n);
		for(i:small=1,M,
			u=lift(vecpol(G[i,],n)*invRnorme);
\\			print(u);
			L[i,]=concat(Mod(polvec(u,m),'x^n+1),[0]);
			for(j:small=1,min(i-1,m),u=sum(z:small=j,m,polconjL(Orth[j,z],n)*L[i,z]);for(z:small=j,m,L[i,z]-=Orth[j,z]*u););
			
			nouvnorm=sum(z:small=i,m,polconjL(L[i,z],n)*L[i,z]); /* carrée ! */
\\			if(i==m,print(i," ",polNorme(lift(nouvnorm),n)^(.5/n));print("L=",lift(L[m,])));
			u=logunit(lift(nouvnorm),n);
\\			print(u);
			for(k:small=1,min(i-1,m),L[i,k]*=u);
			for(k:small=1,m,G[i,k]=(G[i,k]*u)%('x^n+1));

			forstep(j:small=min(i-1,m),1,-1,
				mu=round(lift(L[i,j]*(invLi[j])),&u);
				for(k:small=1,j,L[i,k]-=mu*L[j,k]);
				for(k:small=1,m,G[i,k]-=(mu*G[j,k])%('x^n+1));
			);
			u=lift(vecpol(G[i,],n)*invRnorme);
			L[i,]=concat(Mod(polvec(u,m),'x^n+1),[0]);
			for(j:small=1,min(i-1,m),u=sum(z:small=j,m,polconjL(Orth[j,z],n)*L[i,z]);for(z:small=j,m,L[i,z]-=Orth[j,z]*u););
			if(i>m,
				forstep(j:small=min(i-1,m),1,-1,
					mu=round(lift(L[i,j]*(invLi[j])),&u);
					for(k:small=1,j,L[i,k]-=mu*L[j,k]);
					for(k:small=1,m,G[i,k]-=(mu*G[j,k])%('x^n+1));
				);
				next);
\\			Orth[m,]=L[i,];
\\				print("cur L=",lift(L[i,1..i-1]));
\\				print("fin ",lift(L[i,i..m]));
			for(z:small=i,m,Orth[i,z]=L[i,z]);
			nouvnorm=racine(lift(sum(z:small=i,m,polconjL(L[i,z],n)*L[i,z])),n);
\\			print(i," ",polNorme(lift(nouvnorm),n)^(1/n));
			nouvnorm*=-normalise(lift(L[i,i]),n);
			L[i,i]=nouvnorm;
			invLi[i]=inv(lift(nouvnorm),n);
\\			print("v2 L=",L[i,]);
			Orth[i,i]-=nouvnorm;
			u=inv(lift(racine(lift(sum(z:small=i,m,polconjL(Orth[i,z],n)*Orth[i,z])/2.),n)),n);
\\			print("u=",u);
			for(z:small=i,m,Orth[i,z]*=u);
\\			print("O=",Orth[i,i..m]);
\\			u=sum(z:small=i,m,polconjL(Orth[i,z],n)*Orth[m,z]);for(z:small=i,m,Orth[m,z]-=Orth[i,z]*u);
\\			print("image=",lift(nouvnorm)," vs. ",lift(Orth[m,i..m]));
		);
		print(vecmax(vector(m,i,maxPoly(G[m+1,i])))*1.);
		petitG[1+tour%stock,]=G[m+1,];

\\		if(tour==5,breakpoint);
\\		print(lift(L[m+1,]));
		maxi=1+polNorme(lift(L[1,1]),n)^(1/n);
		mini=if(tour%2==1,min(1,2^-n*polNorme(lift(L[m,m]*L[m-1,m-1]*inv(lift(L[m-2,m-2]),n)),n)^(1/n)),polNorme(lift(L[m,m]),n)^(1/n));
		L[m+1,m+1]=mini;
\\			if(log(maxi/mini)<n^2 && tour%2==0,break);
		curbit[1+tour%2]=truncate(log(maxi/mini)/log(2)*1.3+8*sqrt(n)+1*m*log(n+1)/log(2));
		print(truncate(log(maxi/mini)/log(2)*1.3)," ",mini*1.);
		localprec(truncate(curbit[1+tour%2]*log(2)/log(10)));
\\		if(niveau==2,print(curbit," ",log(maxi)/log(2));print(matrix(m,m,i,j,log(maxPoly(G[i,j]))/log(2))));
\\		for(i=1,m,if(G[i,i]!=polconj(G[i,i],n),print("B ",i);1/0));
		logGS=vector(m,i,abs(polNorme(lift(L[i,i]),n)^(1/n)));
		for(i=1,m,write(fichier,log(logGS[i])/log(2)+1e-3));write(fichier,"\n");
		print(logGS," ",prod(i:small=1,m,logGS[i]));
		if(polNorme(lift(L[1,1]),n)<facteur^n,
			return(matconcat([G;petitG])););
\\		print(matrix(m,m/2,i,j,if(i>j,log(1+maxPoly(lift(L[i,j]*invLi[j])))/log(2),0))~);
		parfor(num:small=0,m\2-1,
			localprec(truncate(curbit[1+tour%2]*log(2)/log(10)-1));
			my(ind,V);
			ind=1+tour%2+2*(m/2-num-1);
			Gm=lift([L[ind,ind],0;L[ind+1,ind],L[ind+1,ind+1]]);
\\			print(ind," ",L[ind+1,ind+1]);
			if(n>16,
				my(fplll:small=0);
				Gm=lift(Gm*invracineC((Gm[1,1]*Gm[2,2])%('x^n+1),n));
				maxi=1+max(max(maxPoly(Gm[1,1]),maxPoly(Gm[2,1])),1);
				mini=min(abs(polNorme(Gm[1,1],n)),abs(polNorme(Gm[2,2],n)))^(1/n);
				bit=-2*log(mini)/log(2);
				f=tailleRecursion(n):small;
				if(bit*10<n*7 && n<=64 && bitprecision(1.)<=(n*m*2)\3,f=max(1,n/16);fplll=1);

\\				print(log(maxi)/log(2)," ",log(mini)/log(2)," ",L[i+1,i+1]);
\\				if(n>30,print(Gm);print(lift(L[i+1,i+1])));
				my(mult,ndeter):real;
				mult=2^(40+6*sqrt(n))/mini;
				ndeter=mult^2;
				Gn=etend(Gm,n,f);
\\				print("récursion");
				Gn=round(Gn*mult,&u);
\\				localprec(truncate((bit+40)*log(2)/log(10)));
\\				print(n," ",f," ",fplll," ",Mod(Gm[1,1],'x^n+1)*mult^2);
				if(!fplll,V=fastlll(Gn,n/f,ndeter,3),
					V=moyenlll(Gn,n/f,3);
					f=n;
					);
				gettime();
				localprec(truncate(curbit[1+tour%2]*log(2)/log(10)-1));
				g=2;
				z=1;
				borne=1;
				iter=0;
				while(g!=1,
					iter++;
					if(iter>2,borne++;iter=0;if(borne>=3,borne=1;z++);/*print(z," ",V)*/);
					vec=vector(2*f,y,if(y<=z,random(2*borne+1)-borne,0))*V;
					A=sum(z:small=0,f-1,multiP(lift(vec[1+z]),n,f)*'x^z);
					B=sum(z:small=0,f-1,multiP(lift(vec[f+1+z]),n,f)*'x^z);
					[C,D,g]=euclideEtendu(A,B,n);
\\					if(n==64,print("j=",j," A=",A," ",B," ",g));
\\					if(g>1 && iter>90 && z==1,print(g," ",vec);if(iter==99,1/0););
					);
				if(ind==m,print(z));
				Gm=lift([L[ind,ind],0;L[ind+1,ind],L[ind+1,ind+1]]);
				Gn=[A,B]*Gm%('x^n+1);
				nouvnorm=sqrt(polNorme(lift(polconjL(Gn[1],n)*Gn[1]+polconjL(Gn[2],n)*Gn[2])%('x^n+1),n));
\\				print(Gn," ",nouvnorm);
				if(nouvnorm>polNorme(lift(L[ind,ind]),n),A=C=1;B=D=0;nouvnorm=polNorme(lift(L[ind,ind]),n));
				if(nouvnorm^2>polNorme(lift(L[ind+1,ind+1]*polconjL(L[ind+1,ind+1],n)+L[ind+1,ind]*polconjL(L[ind+1,ind],n)),n),A=C=0;B=D=1);
				[C,D]=[-D,C];
\\				print("fin: ",[A,B]);
\\				print([C,D]);
				[A,B,C,D]=Mod([A,B,C,D],'x^n+1);
				V=[A,C;B,D];
\\				print("V=",lift(V));
			,
				V=petitlllrapide([L[ind,ind],L[ind+1,ind];0,L[ind+1,ind+1]],n);
\\				print(ind," ",lift(V));
				V=Mod(V,'x^n+1);
				det=lift(V[1,1]*V[2,2]-V[1,2]*V[2,1]);
				V[,1]*=det;
\\				print(ind," ",det);
\\				print("L=",lift([L[ind,ind],L[ind+1,ind];0,L[ind+1,ind+1]]));
\\				print(lift(V));
				);
			V
			,
			V,
			my(i=if(m==2,1,1+tour%2)+2*(m/2-num-1));
			my(det=polNorme(lift(L[i,i]*L[i+1,i+1]),n));
/*			if(i==m,print(det^(1/n)," ",polNorme(lift(L[i,i]*V[1,1]+L[i+1,i]*V[2,1]),n)^(2/n));
			print(V);
			);*/
			if(i<m || det*2^(-n)<polNorme(lift(L[i,i]*V[1,1]+L[i+1,i]*V[2,1]),n)^2,
				for(z:small=1,m,[G[i,z],G[i+1,z]]=lift([G[i,z],G[i+1,z]]*V);););
		);
	);
	U;

}

multiplie(A,normeA,B,normeB,n:small,module:int,fac:int,quotient:small=1)={
\\ multiplie A*B, renvoie [f,g,nor,G] où G=A*B*f/g de norme nor, réduit
	my(norme,m,G,j,petit);
	m=#A;
	norme=(normeA*normeB)%('x^(n*m)+1);
	G=ideauxmul(A,B,norme,n,fac);
	j=1;
	while(gcd(module,if(quotient,polNorme(vecpol(G[j,],n),n*m)^2/polNorme(norme,n*m)))>1,j++);
	petit=vecpol(G[j,],n);
	mul=polconj(petit,n*m)*inv(norme,n*m);
	for(i=1,#G~,G[i,]=polvec(round((vecpol(G[i,],n)*mul)%('x^(n*m)+1)),m));
	[polconj(petit,n*m),norme,lift(round(mul*petit)),G];
}

inverse(pol,module:int)={
	my(f,res,inv,cur);
	f=factor(module);
	res=Mod(0,1);
	for(i=1,#f~,
		cur=Mod(pol,f[i,1]^f[i,2]);
		inv=Mod(liftall(1/Mod(pol,f[i,1])),f[i,1]^f[i,2]);
		for(j=1,f[i,2],inv=inv*(2-inv*cur));
		res=chinese(res,inv));
\\	print(res," ",pol);
	return(res);
}

puissance(A,normeA,n:small,e:int,module:int,fac:int)={
	if(e==1,return([A,normeA,Mod(1,module)]));
	my(f,g,nor,G,B,normeB,produit);
	if(e%2==1,
		[B,normeB,produit]=puissance(A,normeA,n,e-1,module,fac);
		[f,g,nor,G]=multiplie(A,normeA,B,normeB,n,module,fac);
		print("E=",e," !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
		[G,nor,produit*f*inverse(Mod(g,'x^(n*#A)+1),module)];
		,
		[B,normeB,produit]=puissance(A,normeA,n,e/2,module,fac);
\\		print(vecpol(B[1,],n)*polconj(vecpol(B[1,],n),n*#A)*inv(normeB,n*#A));
		[f,g,nor,G]=multiplie(B,normeB,B,normeB,n,module,fac);
		print("E=",e," !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
		[G,nor,produit^2*f*inverse(Mod(g,'x^(n*#A)+1),module)];
		);
}

gentry(A,normeA,n:small,fac:int,fac2:int)={
	my(N,seuil,p,e,q,k,G,nor,produit,norG,den,num);
	my(racine,racineUnite,ok,derive,facteurs,puiss,expo,b,final);
	N=n*#A;
	seuil=fac2*N;
	e=2*N; p=1;
	while (p<seuil,
		e+=2*N;
		p=1;
		fordiv (e, k,
			if ((k%(2*N)==0) && isprime(k+1), p*=(k+1))));
	print(e," ",p);
	q=nextprime(N^2); \\nextprime(1.1^n);\\ si on veut k=1 pour factoriser directement
	while (2*N!=gcd((q^(N/2)-1)*q,e)>30*N || znorder(Mod(q,2*N))<N/2 ,q=nextprime(q+1));
	k=ceil(log(fac2*N^2)/log(q))+4;
	print(q," ",k," exp=",gcd((q^(N/2)-1)*q,e));
	[num,den,norG,G]=multiplie(A,normeA,matrix(1,#A,i,j,j==1),1,n,p*q,fac2); \\ réduction initiale
	[G,nor,produit]=puissance(G,norG,n,e,p*q^k,fac);
	[f,g,nor,G]=multiplie(G,nor,matrix(1,#A,i,j,j==1),1,n,p*q^k,p\N); \\ réduction finale
	produit*=f*inverse(Mod(g,'x^N+1),p*q^k);
	final=centerlift(lift(Mod(produit,p)));
	b=gcdext(e,(q^(N/2)-1)*q^(N/2*(k-1)));
	expo=b[3];
	puiss=(final/Mod(produit,q^k))^(b[1]%((q^(N/2)-1)*q^(N/2*(k-1)))); \\ A^expo mod q^k
	facteurs=lift(factormod('y^N+1,q)[,1]);
	racineN=subst(lift(Mod(puiss,q)),'x,'y);
	while(b[3]%2==0,racineN=lift(polrootsmod('x^2-racineN,[facteurs[1],q])[1]);b[3]/=2);
	racineN=polrootsmod('x^b[3]-racineN,[facteurs[1],q])[1];
	facteurs=subst(Mod(facteurs,q),'y,'x);
	racineN=Mod(Mod(subst(lift(racineN),'y,'x),q),facteurs[1]);
	while(!ok,
		racineUnite=Mod(random(Mod('x^(N/2),q)),facteurs[1])^((q^(N/2)-1)/expo);
		ok=1;
		fordiv(expo,d,if(isprime(d),ok=ok && racineUnite^(expo/d)!=1));
		);
	for(i=1,expo,
		racineN*=racineUnite;
		racine=chinese(racineN,norG/Mod(polconjL(racineN,N),facteurs[2]));
		racine=Mod(Mod(lift(lift(racine)/polcoeff(racine.mod,0)),q^k),'x^N+1);
		derive=inverse(expo*racine^(expo-1),q^k);
		for(i:small=1,k,
			racine-=(racine^expo-puiss)*derive;
			);
		racine=centerlift(lift(racine));
		print(racine);
		if(racine*polconj(racine,N)%('x^N+1)==norG,return(racine*den/Mod(num,'x^N+1)));
	);
}


exec()=\
{
	my(n,t,f,g,h,A);
default(format,"g.5");

n=2^10;
facteur=truncate(2^(n/4));
facteur2=truncate(1.09^n);
t=facteur;
localprec(log(facteur)/log(10)*9);
f=tailleRecursion(n);
g=Pol(vector(n,i,(random(2*t+1)-t)));
print(g);
nor=(g*polconj(g,n))%('x^n+1);
A=matrix(f+4,f);
\\print(t);
for(i=1,f+4,
	h=Pol(vector(n,j,(random(2*t+1)-t)));
	h=(g*h)%('x^n+1);
	A[i,]=polvec(h,f);
	);
\\print(1.*polNorme(g,n)^(tailleRecursion(n)/n));
system(Str("rm ",fichier));
\\multiplie(A,nor,A,nor,n/f,1,facteur);
nouvg=gentry(A,nor,n/f,facteur,facteur2);
}

exportall();
