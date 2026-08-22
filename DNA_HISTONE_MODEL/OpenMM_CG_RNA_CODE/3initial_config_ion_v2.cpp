//******************************************************************
//****     Create initial random positions for ions       ****
//******************************************************************

#include <iostream>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <fstream>
#include <sstream>
#include <string>
#include <random>

using namespace std;

//__ Declaration of the random number generating function between 0 and 1 __//

double rng01();

int main(){

        srand(time(NULL));
        /*N_ion = N_cation + (N_cation - N_phosphate)*/
        long int N_old = 180, N_Mg_new = 10, N_Co_new = 0, N_K_new = 723, N_Cl_new = (2 *N_Co_new) + (2*N_Mg_new) + N_K_new ; 
        /* 
           cMg = 2mM: N_Mg_new = 10, N_Cl_new = 20
           cMg = 4mM: N_Mg_new = 19, N_Cl_new = 38
           cMg = 6mM: N_Mg_new = 29, N_Cl_new = 58
           cMg = 8mM: N_Mg_new = 39, N_Cl_new = 78
           cMg = 8mM: N_Mg_new = 58, N_Cl_new = 116
           cMg =16mM: N_Mg_new = 77, N_Cl_new = 154
           cMg =20mM: N_Mg_new = 96, N_Cl_new = 192
           
        */

	long int N_beads = 180, 
                 N_K     = 0, 
                 N_Mg    = 0,
                 N_Co    = 0,
                 N_Cl    = 0, 
                 N_Phos  = 60;// << for printing final count

        

        long int N = N_old + N_K_new + N_Mg_new + N_Co_new + N_Cl_new - N_Phos;
        cout << N << endl;

        long double L, vol, rxi, ryi, rzi;
        long double Linv, rmin, rminsq, rxij, ryij, rzij, rij, rijsq;
        long double *x,*y,*z;
        long int xyzAtoms;
        char junk[10];

        L = 200;
        Linv = 1.0/L;
      	rmin = 10.0;
        rminsq = rmin*rmin;

        for(int k=1; k <2; k++ ){

                        // int l = 500;
                        string position_file_in, position_file_out;
                        stringstream snap_index;
                        snap_index << k;
                        position_file_in = "./3e5c_cg_new_com.xyz";
                        // position_file_in = "../BasinB/Snap_Mg4_bB_t1_st140083.xyz";
                        // position_file_in = "../BasinC/Snap_Mg4_bC_t1_st172154.xyz";
                        position_file_out = "./3e5c_Mg2.xyz";
                        
                        FILE *initial_position_in, *ion_and_rna_position_xyz;
                        initial_position_in=fopen(position_file_in.c_str(),"r");

                        // FILE *ion_and_rna_position_dat;

                        
                        // ion_and_rna_position_dat=fopen("Snap1500_Mg20mM_K30mM.dat","w");
                        ion_and_rna_position_xyz=fopen(position_file_out.c_str(),"w");
                        fprintf(ion_and_rna_position_xyz,"%ld\n\n",N);
                                
                        x = (long double*)malloc(N*sizeof(long double));
                        y = (long double*)malloc(N*sizeof(long double));
                        z = (long double*)malloc(N*sizeof(long double));
                        // q = (long int*)malloc(N*sizeof(long int));
                        fscanf(initial_position_in,"%ld\n\n",&xyzAtoms);
                        for(int i=0; i < N_old; i++){
                        //  fscanf(initial_position,"%Lf%Lf%Lf%ld",(x+i),(y+i),(z+i),(q+i));
                                fscanf(initial_position_in,"%s%Lf%Lf%Lf",junk,(x+i),(y+i),(z+i));
                        }

                        //__choose next location at random avoiding the overlap__//

                        int i=N_old;
                        int isafe = 0;

                        while(i < N) 
                                { 	        
                                        //__generate a point within the box to place particle i __//
                                        rxi = L*(rng01() - 0.5);
                                        ryi = L*(rng01() - 0.5);
                                        rzi = L*(rng01() - 0.5);

                                        for(int j=0; j < i; j++)
                                        {
                                                
                                                rxij = rxi- *(x+j);
                                                ryij = ryi- *(y+j);
                                                rzij = rzi- *(z+j);

                                                //__Minimum Image Convention__//
                                                rxij = rxij - L*round(rxij*Linv);
                                                ryij = ryij - L*round(ryij*Linv);
                                                rzij = rzij - L*round(rzij*Linv);
                                                
                                                rijsq =(rxij*rxij + ryij*ryij + rzij*rzij);
                                        //	rij = sqrt(rijsq);
                                                if (rijsq > rminsq)
                                                {
                                                        isafe = 1;
                                                //	cout << sqrt(rijsq) << endl;
                                                }
                                                else
                                                {
                                                        isafe= 0;
                                                        break;	
                                                }
                                                

                                        }

                                        if(isafe == 1)
                                        {
                                                        
                                        *(x+i) = rxi;
                                        *(y+i) = ryi;
                                        *(z+i) = rzi;
                                        
                                        i++;
                                        }
                                        
                                }

                        // for(int i=0; i<N; i++){fprintf(ion_and_rna_position_dat,"%Lf\t%Lf\t%Lf\n",*(x+i),*(y+i),*(z+i));}
                        
                        for(int i=0; i<N_beads; i++){
                                if(i%3==0){
                                fprintf(ion_and_rna_position_xyz,"P\t%Lf\t%Lf\t%Lf\n",*(x+i),*(y+i),*(z+i));
                                }
                                else if(i%3==1){
                                fprintf(ion_and_rna_position_xyz,"S\t%Lf\t%Lf\t%Lf\n",*(x+i),*(y+i),*(z+i));
                                }
                                else if(i%3==2){
                                fprintf(ion_and_rna_position_xyz,"B\t%Lf\t%Lf\t%Lf\n",*(x+i),*(y+i),*(z+i));
                                }
                        }
                        for (int i= N_beads; i < (N_beads + N_Mg); i++){ // printing old Mg2+ ions
                                fprintf(ion_and_rna_position_xyz,"Mg\t%Lf\t%Lf\t%Lf\n",*(x+i),*(y+i),*(z+i));
                        } 
                        for (int i= N_old; i < (N_old + N_Mg_new); i++){// printing new Mg2+ ions
                                fprintf(ion_and_rna_position_xyz,"Mg\t%Lf\t%Lf\t%Lf\n",*(x+i),*(y+i),*(z+i));
                        }
                        for (int i= (N_beads+ N_Mg); i < (N_beads + N_Mg + N_Co); i++){ // printing old Co2+ ions
                                fprintf(ion_and_rna_position_xyz,"Co\t%Lf\t%Lf\t%Lf\n",*(x+i),*(y+i),*(z+i));
                        } 
                        for (int i= (N_old+ N_Mg_new); i < (N_old + N_Mg_new + N_Co_new); i++){// printing new Co2+ ions
                                fprintf(ion_and_rna_position_xyz,"Co\t%Lf\t%Lf\t%Lf\n",*(x+i),*(y+i),*(z+i));
                        }
                        for (int i= (N_beads+N_Mg+ N_Co); i < (N_beads + N_Mg + N_Co + N_K); i++){// printing old K+ ions
                                fprintf(ion_and_rna_position_xyz,"K\t%Lf\t%Lf\t%Lf\n",*(x+i),*(y+i),*(z+i));
                        }
                        for (int i= (N_old+ N_Mg_new+ N_Co_new); i < (N_old + N_Mg_new + N_Co_new + N_K_new); i++){// printing new K+ ions
                                fprintf(ion_and_rna_position_xyz,"K\t%Lf\t%Lf\t%Lf\n",*(x+i),*(y+i),*(z+i));
                        }
                        for(i= (N_beads + N_Mg+ N_Co + N_K ); i < N_old; i++){// printing old CL- ions
                                fprintf(ion_and_rna_position_xyz,"Cl\t%Lf\t%Lf\t%Lf\n",*(x+i),*(y+i),*(z+i));
                        }        
                        for(i= (N_old + N_Mg_new + N_Co_new + N_K_new); i < N; i++){// printing new CL- ions
                                fprintf(ion_and_rna_position_xyz,"Cl\t%Lf\t%Lf\t%Lf\n",*(x+i),*(y+i),*(z+i));
                        }
                                
                // close the opened files.
                fclose(initial_position_in);
                // fclose(ion_and_rna_position_dat);
                fclose(ion_and_rna_position_xyz);
        
        }

return 0;

}


double rng01()
{
        double r = ((double) rand())/RAND_MAX;
        return(r);
}


