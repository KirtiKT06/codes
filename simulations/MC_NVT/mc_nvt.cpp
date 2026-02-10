// ============================================================
// NVT Monte Carlo simulation of Lennard-Jones fluid
// Implements:
// - Pressure and energy (virial + tail corrections)
// - Widom insertion for excess chemical potential
// - Equation of state generation for thermodunamic integration
// ============================================================
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <random>
#include <numbers>
#include <chrono>
#include <algorithm>
#include <numeric>      // for std::iota
#include <string>       // forstd::string

// ------------------------------------------------------------
// Simple structure to store (T, rho) state points
// ------------------------------------------------------------
struct StatePoint 
{
    double T;
    double rho;
};

// ------------------------------------------------------------
// Container for averaged Monte Carlo results
// ------------------------------------------------------------
struct mc_results
{
    double P_avg;        // average pressure
    double U_avg;        // average energy per particle
    double beta_mu;      // Widom excess chemical potential
    int widom_samples;   // total number of Widom insertions
};

// ============================================================
// Main Monte Carlo class (NVT ensemble)
// ============================================================
class mc_nvt
{
    public:
    // -------------------------
    // System parameters
    // -------------------------
    int N;                  // number of particles
    double box_len;         // simulation box length
    double temp;            // reduced temprature
    double beta;            // inverse temperature (1/T)
    double u_total;         // total potential energy 
    double rho;             // reduced density
    double delta;           // MC displacement amplitude

    double r_cut = 2.5;     // LJ cutoff radius
    double r_cut_sq;        // squared cutoff radius

    // -------------------------
    // Neighbor list parameters
    // -------------------------
    std::vector<std::vector<int>> neigh;    // verlet neighbour list
    double r_list;                          // neighbour list cutoff
    double r_list_sq;                       // squared neighbour list cutoff
    double skin;                            // skin distance

    // -------------------------
    // Particle coordinates
    // -------------------------
    std::vector <double> x, y, z;
    // Accumulated displacements (for neighbor list rebuild)
    std::vector<double> dx_acc, dy_acc, dz_acc;

    // Randomized particle index order
    std::vector<int> indices;

    // Random number generator
    std::mt19937 rng;
    std::uniform_real_distribution<double> uni01;

    // ========================================================
    // Initialize particle positions on a cubic lattice
    // ========================================================
    void init_config()
    {
        beta = 1.0/temp;
        box_len = cbrt(N/rho);

        // Allocate coordinate arrays
        x.resize(N);
        y.resize(N);
        z.resize(N);

        // Reset accumulated displacements
        dx_acc.assign(N, 0.0);
        dy_acc.assign(N, 0.0);
        dz_acc.assign(N, 0.0);

        // Simple cubic lattice initialization
        int n_cell = ceil(cbrt(N));
        double a = box_len/n_cell;

        u_total = 0.0;

        int i = 0;
        for(int ix=0; ix<n_cell; ix++)
        {
            for(int iy=0; iy<n_cell; iy++)
            {
                for(int iz=0; iz<n_cell; iz++)
                {
                    if (i < N)
                    {
                        x[i] = (ix + 0.5)*a;
                        y[i] = (iy + 0.5)*a;
                        z[i] = (iz + 0.5)*a;
                        i++;
                    }
                }
            }
        }

        // Compute initial total energy by direct summation
        u_total = 0.0;
        for (int i=0; i<N-1; i++)
            for (int j=i+1; j<N; j++)
                u_total += pair_energy(i,j);
    }

    // ========================================================
    // Build Verlet neighbor list using minimum-image convention
    // ========================================================
    void build_neighbor_list()
    {
        neigh.assign(N, {});

        for (int i = 0; i < N - 1; i++)
        {
            for (int j = i + 1; j < N; j++)
            {
                double dx = x[i] - x[j];
                dx -= box_len * std::round(dx / box_len);
                double dy = y[i] - y[j];
                dy -= box_len * std::round(dy / box_len);
                double dz = z[i] - z[j];
                dz -= box_len * std::round(dz / box_len);

                double r2 = dx*dx + dy*dy + dz*dz;
                if (r2 < r_list_sq)
                {
                    neigh[i].push_back(j);
                    neigh[j].push_back(i);
                }
            }
        }
    }

    // ========================================================
    // Check whether neighbor list needs rebuilding
    // ========================================================
    bool need_rebuild() const
    {
        double max_disp_sq = 0.0;
        for(int i=0; i<N; i++)
        {
            double d_sq = dx_acc[i] * dx_acc[i]
                + dy_acc[i] * dy_acc[i]
                + dz_acc[i] * dz_acc[i];
            max_disp_sq = std::max(max_disp_sq, d_sq);
        }
        return max_disp_sq > 0.25 * skin * skin;
    }

    // ========================================================
    // Lennard-Jones pair potential (truncated)
    // ========================================================
    inline double pair_energy(int i, int j) const
    {
        double dx = x[i] - x[j];
        double dy = y[i] - y[j];
        double dz = z[i] - z[j];

        dx -= box_len * std::round(dx / box_len);
        dy -= box_len * std::round(dy / box_len);
        dz -= box_len * std::round(dz / box_len);

        double r2 = dx * dx + dy * dy + dz * dz;
        if (r2 > r_cut_sq) return 0.0;

        double inv_r2 = 1.0 / r2;
        double sr6 = inv_r2 * inv_r2 * inv_r2;
        double sr12 = sr6 * sr6;
        
        return 4.0 * (sr12 - sr6);
    }

    // ========================================================
    // Attempt a Monte Carlo move for particle i
    // ========================================================
    bool particle_move(int i)                               
    {
        // Save old position
        double x_old = x[i];
        double y_old = y[i];
        double z_old = z[i];

        // Random displacement
        double dx = delta * (uni01(rng) - 0.5);
        double dy = delta * (uni01(rng) - 0.5);
        double dz = delta * (uni01(rng) - 0.5);

        // Energy before move
        double delta_u = 0.0;

        for(int j : neigh[i])
            delta_u -= pair_energy(i, j); 

        // Apply displacement with periodic boundaries
        x[i] += dx;
        x[i] -= box_len * (double) std::floor(x[i]/box_len);
        y[i] += dy;
        y[i] -= box_len * (double) std::floor(y[i]/box_len);
        z[i] += dz;  
        z[i] -= box_len * (double) std::floor(z[i]/box_len);

        // Energy after move
        for(int j : neigh[i])
            delta_u += pair_energy(i,j);

        // Metropolis acceptance
        if (delta_u <= 0.0 || uni01(rng) < std::exp(-beta*delta_u))
        {
            u_total += delta_u;

            dx_acc[i] += dx;
            dy_acc[i] += dy;
            dz_acc[i] += dz;

            return true;
        }
        else
        {
            // Reject move
            x[i] = x_old;
            y[i] = y_old;
            z[i] = z_old;
            return false; 
        }
    }

    // ========================================================
    // Perform one Monte Carlo cycle (N attempted moves)
    // ========================================================
    double mc_cycle()
    {
        if (need_rebuild())
        {
            build_neighbor_list();
            reset_displacements();
        }

        std::shuffle(indices.begin(), indices.end(), rng);

        int accepted = 0;
        for (int k : indices)
        {
            if (particle_move(k))
                accepted++;
        }
        return static_cast<double>(accepted) / N;
    }

    // Reset accumulated displacements
    inline void reset_displacements()
    {
        std::fill(dx_acc.begin(), dx_acc.end(), 0.0);
        std::fill(dy_acc.begin(), dy_acc.end(), 0.0);
        std::fill(dz_acc.begin(), dz_acc.end(), 0.0);
    }

    // ========================================================
    // Compute virial contribution to pressure
    // ========================================================
    void compute_virial_only(double& w)
    {
        w = 0.0;
        for (int i = 0; i < N; i++)
        {
            for (int j : neigh[i])
            {
                if (j <= i) continue;

                double dx = x[i] - x[j];
                dx -= box_len * std::round(dx / box_len);
                double dy = y[i] - y[j];
                dy -= box_len * std::round(dy / box_len);
                double dz = z[i] - z[j];
                dz -= box_len * std::round(dz / box_len);

                double r2 = dx*dx + dy*dy + dz*dz;
                if (r2 > r_cut_sq) continue;

                double inv_r2 = 1.0 / r2;
                double sr6 = inv_r2 * inv_r2 * inv_r2;
                double sr12 = sr6 * sr6;
                w += 24.0 * (2.0 * sr12 - sr6);
            }
        }
    }

    // ========================================================
    // Widom test particle insertion energy
    // ========================================================
    double widom_insertion_energy()
    {
        double xt = uni01(rng) * box_len;
        double yt = uni01(rng) * box_len;
        double zt = uni01(rng) * box_len;

        double deltaU = 0.0;

        for (int i = 0; i < N; i++)
        {
            double dx = x[i] - xt;
            double dy = y[i] - yt;
            double dz = z[i] - zt;

            dx -= box_len * std::round(dx / box_len);
            dy -= box_len * std::round(dy / box_len);
            dz -= box_len * std::round(dz / box_len);

            double r2 = dx*dx + dy*dy + dz*dz;

            if (r2 < r_cut_sq)
            {
                double inv_r2 = 1.0 / r2;
                double sr6 = inv_r2 * inv_r2 * inv_r2;
                double sr12 = sr6 * sr6;
                deltaU += 4.0 * (sr12 - sr6);
            }
        }

        return deltaU;
    }

    // ========================================================
    // Run a single thermodynamic state point
    // ========================================================
    mc_results run_single_state(
        double T_in,
        double rho_in,
        int N_in,
        bool do_timeseries,
        bool do_widom,
        int seed,
        std::ostream* traj_stream = nullptr)
    {
        auto t0 = std::chrono::high_resolution_clock::now();

        // -----------------------------
        // Basic thermodynamic setup
        // -----------------------------
        N    = N_in;
        temp = T_in;
        rho  = rho_in;
        beta = 1.0 / temp;

        indices.resize(N);
        std::iota(indices.begin(), indices.end(), 0);

        delta = 0.1;
        r_cut = (rho < 0.5 ? 2.5 : 3.0);
        r_cut_sq = r_cut * r_cut;

        // LJ shift
        double inv_rc2 = 1.0 / r_cut_sq;
        double sr6_rc  = inv_rc2 * inv_rc2 * inv_rc2;
        double sr12_rc = sr6_rc * sr6_rc;

        r_list = r_cut + 0.4;
        r_list_sq = r_list * r_list;
        skin = r_list - r_cut;

        std::cout << "\n----------------------------------------\n";
        std::cout << "Running state:\n";
        std::cout << "T* = " << T_in << ", rho* = " << rho_in << "\n";
        std::cout << "N = " << N_in << "\n";
        std::cout << "r_cut = " << r_cut << ", r_list = " << r_list << "\n";
        std::cout << "----------------------------------------\n";

        rng.seed(seed);
        uni01 = std::uniform_real_distribution<double>(0.0, 1.0);

        // Tail corrections
        const double rc3 = r_cut * r_cut * r_cut;
        const double rc9 = rc3 * rc3 * rc3;

        const double p_tail_const =
            (16.0 * std::numbers::pi / 3.0) * rho * rho *
            ((2.0 / (3.0 * rc9)) - (1.0 / rc3));

        const double u_tail_per_particle =
            (8.0 * std::numbers::pi * rho / 3.0) *
            ((1.0 / (3.0 * rc9)) - (1.0 / rc3));
        
        // ---- Widom-specific long-range correction ----
        // Effective density seen by test particle (N-1 particles in volume V)
        double rho_eff = rho * (N - 1.0) / N;

        // Tail correction for excess chemical potential (Widom insertion)
        const double mu_tail_widom =
            (8.0 * std::numbers::pi * rho_eff / 3.0) *
            ((1.0 / (3.0 * rc9)) - (1.0 / rc3));
            
        // -----------------------------
        // Initialization
        // -----------------------------
        init_config();
        build_neighbor_list();

        // -----------------------------
        // Equilibration
        // -----------------------------
        int n_equil = (rho < 0.5 ? 20000 : 100000);
        double target = (rho < 0.05 ? 0.6 : 0.35);

        for (int i = 0; i < n_equil; i++)
        {
            double acc_rate = mc_cycle();

            if (i % 20 == 0)
            {
                delta *= std::exp(0.01 * (acc_rate - target));
                delta = std::clamp(delta, 0.01, 0.5 * skin);
            }
        }

        std::cout << "Equilibration done.\n";
        std::cout << "Final delta = " << delta << "\n";

        // -----------------------------
        // Production
        // -----------------------------
        int n_production = (rho < 0.6 ? 100000 : 1000000);
        int sample_stride = (rho < 0.5 ? 200 : 100);
        int widom_stride  = 5 * sample_stride;
        int insertion_per_config = rho > 0.6 ? 5000 : 1000;

        double u_sum = 0.0, p_sum = 0.0;
        int n_samples = 0;

        double widom_sum = 0.0;
        int widom_count = 0;
        int neg_count = 0;

        for (int i = 0; i < n_production; i++)
        {
            
            if (i % (n_production / 10) == 0 && i > 0)
            {
                std::cout << "Production progress: "
                << (100.0 * i / n_production) << "%\n";
            }

            mc_cycle();

            // ---- Thermodynamic sampling
            if (i % sample_stride == 0)
            {
                double w;
                compute_virial_only(w);
                double V = box_len * box_len * box_len;

                double p_inst = rho * temp + w / (3.0 * V);
                p_inst += p_tail_const;

                double u_inst = u_total + N * u_tail_per_particle;

                u_sum += u_inst;
                p_sum += p_inst;
                n_samples++;

                if (do_timeseries && traj_stream)
                {
                    (*traj_stream) << i << " " << p_inst << " " << u_inst << "\n";
                }
            }

            // ---- Widom insertion
            if (do_widom && i % widom_stride == 0)
            {
                for (int k = 0; k < insertion_per_config; k++)
                {
                    double dU = widom_insertion_energy();
                    widom_sum += std::exp(-beta * dU);
                    widom_count++;
                            if (dU < 0.0)
                                neg_count++;
                }
            }
        }

        std::cout << "Widom diagnostics:\n";
        std::cout << "Total Widom insertions = " << widom_count << "\n";
        std::cout << "<exp(-beta ΔU)> = "
            << (widom_sum / widom_count) << "\n";
        
        std::cout << "Fraction of ΔU < 0: "
            << double(neg_count) / widom_count << "\n";

        // -----------------------------
        // Final averages
        // -----------------------------
        mc_results result;
        result.P_avg = p_sum / n_samples;
        result.U_avg = (u_sum / n_samples) / N;
        result.beta_mu =
            do_widom
            ? -std::log(widom_sum / widom_count) + beta * mu_tail_widom : NAN;
        result.widom_samples = widom_count;

        auto t1 = std::chrono::high_resolution_clock::now();
        std::cout << "Runtime (s) = "
                << std::chrono::duration<double>(t1 - t0).count()
                << std::endl;

        return result;
    }
};

int main(int argc, char* argv[])
{
   if (argc < 2)
    {
        std::cerr << "Usage: ./mc_nvt MODE\n";
        std::cerr << "MODE = 1 (Q1), 2 (Q2), 3 (Q3)\n";
        return 1;
    }

    std::cout.setf(std::ios::unitbuf);
    mc_nvt sim;

    // ----------------------------------------
    // MODE SELECTOR
    // 1 = Q1 : Block averaging (time series)
    // 2 = Q2 : Widom insertion
    // 3 = Q3 : EOS / isotherms
    // ----------------------------------------
    int MODE = std::stoi(argv[1]);

    std::cout << "STARTING SIMULATION, MODE = " << MODE << "\n";

    if (MODE == 1)
    {
        // ==============================
        // Q1: Block averaging
        // ==============================
        std::vector<StatePoint> states = {
            {0.8, 0.005},
            {0.9, 0.01},
            {0.9, 0.77},
            {1.0, 0.02},
            {1.0, 0.75},
            {2.0, 0.80}
        };
        int rep = 6;

        for (const auto& s : states)
        {
            for (int k=0; k<rep; k++)
            {
                std::string fname =
                "timeseries_T" + std::to_string(s.T) +
                "_rho" + std::to_string(s.rho) +
                "_rep" + std::to_string(k) + ".dat";

                std::ofstream traj(fname);
                traj << "# Step  P_inst  U_inst\n";

                auto res = sim.run_single_state(
                s.T, s.rho, 256,
                true,   // write time series
                false,  // no Widom
                1234 + k,
                &traj);
            
                traj.close();

                std::cout << "# T*  rho*  P*  U*/N\n";
                std::cout << s.T << "\t" << s.rho << "\t"
                      << res.P_avg << "\t"
                      << res.U_avg << "\n";
            }
        }
    }
    else if (MODE == 2)
    {
        // ==============================
        // Q2: Widom insertion (replicas)
        // ==============================
        std::vector<StatePoint> states = {
            {2.0, 0.5},
            {2.0, 0.9}
        };

        int n_replica = 5;
        std::ofstream widom_out("widom_results.dat");
        widom_out << "# T*  rho*  beta_mu  widom_samples  seed\n";

        for (const auto& s : states)
        {
            for (int k = 0; k < n_replica; k++)
            {
                int seed = 1234 + 100 * k;

                auto res = sim.run_single_state(
                    s.T, s.rho, 256,
                    false,  // no time series
                    true,   // Widom ON
                    seed
                );

                // write to file
                widom_out << s.T << " "
                        << s.rho << " "
                        << res.beta_mu << " "
                        << res.widom_samples << " "
                        << seed << "\n";

                // print to terminal
                std::cout << "# T*  rho*  beta_mu  widom_samples\n";
                std::cout << s.T << "\t"
                        << s.rho << "\t"
                        << res.beta_mu << "\t"
                        << res.widom_samples << "\n";
            }
        }
        widom_out.close();
    }
    else if (MODE == 3)
    {
        // ==============================
        // Q3: EOS / isotherm (data dump)
        // ==============================
        double T = 2.0;
        std::vector<double> rhos = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9};
        std::string fname = "eos_T" + std::to_string(T) + ".dat";
        std::ofstream eos_out(fname);
        eos_out << "# T*   rho*    P*        U*/N\n";

        for (double rho : rhos)
        {
            auto res = sim.run_single_state(
                T, rho, 256,
                false,   // no time series
                false,   // no Widom
                1234
            );

            // write to file (for Python)
            eos_out << T << " "
                    << rho << " "
                    << res.P_avg << " "
                    << res.U_avg << "\n";

            // also print to terminal
            std::cout << "# T*  rho*  P*  U*/N\n";
            std::cout << T << "\t" << rho << "\t"
                    << res.P_avg << "\t"
                    << res.U_avg << "\n";
        }
        eos_out.close();
    }
    else
    {
        std::cerr << "ERROR: Unknown MODE\n";
    }
    return 0;
}