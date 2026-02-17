// ============================================================
// Grand Canonical Monte Carlo (μVT) simulation of a Lennard-Jones fluid
// Features:
// - Particle insertion, deletion, and displacement moves
// - Metropolis acceptance in μVT ensemble
// - Verlet neighbor lists with displacement-based rebuild criterion
// - Replica averaging for statistical error estimation
// ============================================================
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <random>
#include <numbers>
#include <algorithm>
#include <numeric>      // for std::iota
#include <string>       // forstd::string

// ------------------------------------------------------------
// Simple structure to store (T, rho) state points
// ------------------------------------------------------------
struct GCMC_state
{
    double T;
    double f;
};

// ============================================================
// Main Monte Carlo class (μVT ensemble)
// ============================================================
class gcmc
{
    public:
    // -------------------------
    // Geometric parameters
    // -------------------------
    int dim;                // box dimension
    double box_len;         // simulation box length
    double box_volume;      // volume of the simulation box

    // -------------------------
    // Thermodynamic parameters
    // -------------------------
    double temp;            // reduced temprature
    double beta;            // inverse temperature (1/T)
    double f;               // fugacity 

    // -------------------------
    // Interaction parameters
    // -------------------------
    double r_cut = 2.5;     // LJ cutoff radius
    double r_cut_sq;        // squared cutoff radius
    double r_cut_shift;     // shifted LJ potential at r_cut

    // -------------------------
    // MC control parameters
    // -------------------------
    double p_insert;        // probability of insertion of particle
    double p_delete;        // probability of deletion of particle
    double p_move;          // probability of translational motion
    double delta;           // MC displacement amplitude 
    int move_attempts = 0;
    int move_accepts  = 0;

    // Random number generator
    std::mt19937 rng;
    std::uniform_real_distribution<double> uni01;

    // -------------------------
    // Dynamic parameters
    // -------------------------
    int n;                          // number of particles
    double u_total;                 // total potential energy 
    std::vector <double> x, y, z;   // particle coordinates

    // -------------------------
    // Neighbor list parameters
    // -------------------------
    std::vector<std::vector<int>> neigh;    // verlet neighbour list
    double r_list;                          // neighbour list cutoff
    double r_list_sq;                       // squared neighbour list cutoff
    double skin;                            // skin distance

    // Accumulated displacements (for neighbor list rebuild)
    std::vector<double> dx_acc, dy_acc, dz_acc;

    // ========================================================
    // Initialization of variables
    // ========================================================
    void initialize(
    double T_in,
    double f_in,
    double L_in,
    int seed = 1234)
    {
        // Geometry
        dim = 3;
        box_len = L_in;
        box_volume = box_len * box_len * box_len;

        // Thermodynamics
        temp = T_in;
        beta = 1.0 / temp;
        f = f_in;

        // Interaction
        r_cut_sq = r_cut * r_cut;
        double inv_rc2 = 1.0 / r_cut_sq;
        double sr6_rc  = inv_rc2 * inv_rc2 * inv_rc2;
        double sr12_rc = sr6_rc * sr6_rc;
        r_cut_shift = 4.0 * (sr12_rc - sr6_rc);

        // MC control
        p_insert = (f < 0.5 ? 0.30 : 0.40);
        p_delete = (f < 0.5 ? 0.20 : 0.40);
        p_move   = (f < 0.5 ? 0.50 : 0.20);   
        delta    = 0.10;

        // Dynamic state
        n = 0;
        u_total = 0.0;
        x.clear(); y.clear(); z.clear();

        // RNG
        rng.seed(seed);
        uni01 = std::uniform_real_distribution<double>(0.0, 1.0);

        // Neighbour list        
        skin = 0.4;
        r_list = r_cut + skin;
        r_list_sq = r_list * r_list;        
        neigh.clear();
        dx_acc.clear();
        dy_acc.clear();
        dz_acc.clear();

        // Probability constraint
        if (std::abs(p_insert + p_delete + p_move - 1.0) > 1e-12)
        {
            std::cerr << "ERROR: Move probabilities do not sum to 1\n";
            std::exit(1);
        }
    }

    // ========================================================
    // Build Verlet neighbor list using minimum-image convention
    // ========================================================
    void build_neighbor_list()
    {
        neigh.assign(n, std::vector<int>());
        for (int i = 0; i < n - 1; i++)
        {
            for (int j = i + 1; j < n; j++)
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
    bool need_rebuild()
    {
        if (n == 0) return false;

        double max_disp_sq = 0.0;

        for (int i = 0; i < n; i++)
        {
            double d2 = dx_acc[i]*dx_acc[i]
                    + dy_acc[i]*dy_acc[i]
                    + dz_acc[i]*dz_acc[i];
            max_disp_sq = std::max(max_disp_sq, d2);
        }

        return max_disp_sq > 0.25 * skin * skin;
    }

    void reset_displacements()
    {
        std::fill(dx_acc.begin(), dx_acc.end(), 0.0);
        std::fill(dy_acc.begin(), dy_acc.end(), 0.0);
        std::fill(dz_acc.begin(), dz_acc.end(), 0.0);
    }

    // ========================================================
    // Lennard-Jones potential before and after move
    // ========================================================
    double deltaU_move(int i, double x_new, double y_new, double z_new)
    {
        double delta_u = 0.0;

        for (int j : neigh[i])
        {
            // old position
            double dx = x[i] - x[j];
            dx -= box_len * std::round(dx / box_len);
            double dy = y[i] - y[j];
            dy -= box_len * std::round(dy / box_len);
            double dz = z[i] - z[j];
            dz -= box_len * std::round(dz / box_len);

            double r2 = dx*dx + dy*dy + dz*dz;
            if (r2 < r_cut_sq)
            {
                double inv = 1.0 / r2;
                double sr6 = inv*inv*inv;
                double sr12 = sr6*sr6;
                delta_u -= 4.0*(sr12 - sr6) - r_cut_shift;
            }

            // new position
            dx = x_new - x[j];
            dx -= box_len * std::round(dx / box_len);
            dy = y_new - y[j];
            dy -= box_len * std::round(dy / box_len);
            dz = z_new - z[j];
            dz -= box_len * std::round(dz / box_len);

            r2 = dx*dx + dy*dy + dz*dz;
            if (r2 < r_cut_sq)
            {
                double inv = 1.0 / r2;
                double sr6 = inv*inv*inv;
                double sr12 = sr6*sr6;
                delta_u += 4.0*(sr12 - sr6) - r_cut_shift;
            }
        }

        return delta_u;
    }

    // ========================================================
    // Attempt a Monte Carlo move for particle i
    // ========================================================
    bool attempt_move()
    {
        if (n == 0) return false;

        std::uniform_int_distribution<int> pick(0, n - 1);
        int i = pick(rng);

        double x_old = x[i];
        double y_old = y[i];
        double z_old = z[i];

        double dx = delta * (uni01(rng) - 0.5);
        double dy = delta * (uni01(rng) - 0.5);
        double dz = delta * (uni01(rng) - 0.5);

        double x_new = x_old + dx;
        double y_new = y_old + dy;
        double z_new = z_old + dz;

        // periodic boundaries
        x_new -= box_len * std::floor(x_new / box_len);
        y_new -= box_len * std::floor(y_new / box_len);
        z_new -= box_len * std::floor(z_new / box_len);

        double delta_u = deltaU_move(i, x_new, y_new, z_new);

        if (delta_u <= 0.0 || uni01(rng) < std::exp(-beta * delta_u))
        {
            x[i] = x_new;
            y[i] = y_new;
            z[i] = z_new;

            u_total += delta_u;

            dx_acc[i] += dx;
            dy_acc[i] += dy;
            dz_acc[i] += dz;

            return true;
        }

        return false;
    }

    // ========================================================
    // Perform a single Monte Carlo step by selecting one of:
    // - particle insertion
    // - particle deletion
    // - particle displacement
    // Move type is chosen according to fixed probabilities.
    // ========================================================
    void mc_step()
    {
        if (need_rebuild())
        {
            build_neighbor_list();
            reset_displacements();
        }
        
        double r = uni01(rng);

        if (r < p_insert)
            attempt_insert();
        else if (r < p_insert + p_delete)
            attempt_delete();
        else
        {
            move_attempts++;
            if (attempt_move())
                move_accepts++;
        }
    }

    // ========================================================
    // Insertion Move
    // ========================================================
    double deltaU_insert(double x_new, double y_new, double z_new)
    {
        double delta_u = 0.0;
        for(int i=0; i<n; i++)
        {
            double dx = x[i] - x_new;
            dx -= box_len * std::round(dx / box_len);
            double dy = y[i] - y_new;
            dy -= box_len * std::round(dy / box_len);
            double dz = z[i] - z_new;
            dz -= box_len * std::round(dz / box_len);

            double r_squared = dx*dx + dy*dy + dz*dz;
            if (r_squared < r_cut_sq)
            {
            double inv_r2 = 1.0 / r_squared;
            double sr6 = inv_r2 * inv_r2 * inv_r2;
            double sr12 = sr6 * sr6;

            // shifted Lennard–Jones
            delta_u += 4.0 * (sr12 - sr6) - r_cut_shift;
            }
        }
        return delta_u;
    }

    // ========================================================
    // Deletion Move
    // ========================================================
    double deltaU_delete(int k)
    {
        double delta_u = 0.0;
        for (int i = 0; i < n; i++)
        {
            if (i == k) continue;

            double dx = x[i] - x[k];
            dx -= box_len * std::round(dx / box_len);
            double dy = y[i] - y[k];
            dy -= box_len * std::round(dy / box_len);
            double dz = z[i] - z[k];
            dz -= box_len * std::round(dz / box_len);

            double r2 = dx*dx + dy*dy + dz*dz;
            if (r2 < r_cut_sq)
            {
                double inv_r2 = 1.0 / r2;
                double sr6 = inv_r2 * inv_r2 * inv_r2;
                double sr12 = sr6 * sr6;

                // same shifted LJ as insertion
                delta_u += 4.0 * (sr12 - sr6) - r_cut_shift;
            }
        }

        return delta_u;
    }

    // ========================================================
    // Attempt insertion
    // ========================================================
    bool attempt_insert()
    {
        // 1. Trial position (uniform in box)
        double x_new = uni01(rng) * box_len;
        double y_new = uni01(rng) * box_len;
        double z_new = uni01(rng) * box_len;

        // 2. Energy change
        double delta_u = deltaU_insert(x_new, y_new, z_new);

        // 3. Acceptance probability
        double prob = (f * box_volume / (n + 1)) * std::exp(-beta * delta_u);

        prob = std::min(1.0, prob);

        // 4. Accept / reject
        if (uni01(rng) < prob)
        {
            // 1. Add particle
            x.push_back(x_new);
            y.push_back(y_new);
            z.push_back(z_new);

            n++;
            u_total += delta_u;

            // 2. Neighbor-list bookkeeping (THIS IS WHAT YOU ASKED ABOUT)
            dx_acc.push_back(0.0);
            dy_acc.push_back(0.0);
            dz_acc.push_back(0.0);

            build_neighbor_list();
            reset_displacements();

            return true;
        }

        return false;
    }

    // ========================================================
    // Attempt deletion
    // ========================================================
    bool attempt_delete()
    {
        // Cannot delete if system is empty
        if (n == 0) return false;

        // 1. Pick random particle
        std::uniform_int_distribution<int> pick(0, n - 1);
        int k = pick(rng);

        // 2. Energy change
        double delta_u = deltaU_delete(k);

        // 3. Acceptance probability
        double prob = (n / (f * box_volume)) * std::exp(beta * delta_u);
        prob = std::min(1.0, prob);

        // 4. Accept / reject
        if (uni01(rng) < prob)
        {
            // 1. Remove particle (swap with last)
            x[k] = x.back();  x.pop_back();
            y[k] = y.back();  y.pop_back();
            z[k] = z.back();  z.pop_back();

            // 2. Keep displacement arrays consistent
            dx_acc[k] = dx_acc.back(); dx_acc.pop_back();
            dy_acc[k] = dy_acc.back(); dy_acc.pop_back();
            dz_acc[k] = dz_acc.back(); dz_acc.pop_back();

            n--;
            u_total -= delta_u;

            // 3. Rebuild neighbor list
            build_neighbor_list();
            reset_displacements();

            return true;
        }

        return false;
    }

};

// ========================================================
// Main
// ========================================================
int main()
{
    // -----------------------------
    // Simulation parameters
    // -----------------------------
    double T    = 1.0;              // reduced temperature
    double f    = 0.0365;           // fugacity
    double L    = 7.0;              // box length
    int seed    = 12345;            // initial seed

    int n_equil = 200000;           // equilibration steps
    int n_prod  = 2000000;          // production steps
    int stride  = 50;               // sampling interval

    // -----------------------------
    // Thermodynamic state points
    // -----------------------------
    std::vector<GCMC_state> states = {
        {1.0, 0.0365},
        {2.0, 0.767}
        };
    
    std::cout.setf(std::ios::unitbuf);
    std::cout << "STARTING GCMC SIMULATIONS\n";

    int n_replicas = 5;

    for (const auto& s : states)
    {
        std::vector<double> rho_rep;
        std::vector<double> N_rep;

        std::cout << "\n========================================\n";
        std::cout << "Running state: T* = " << s.T
                  << " , f = " << s.f << "\n";
        std::cout << "========================================\n";

        // -----------------------------
        // Average observables over independent replicas.
        // Reported errors correspond to the standard error of the mean.
        // -----------------------------

        for(int r=0; r<n_replicas; r++)
        {
            // -----------------------------
            // Initialize simulation
            // -----------------------------

            int seed_r = seed + 1000 * r;
            gcmc sim;
            sim.initialize(s.T, s.f, L, seed_r);

            // -----------------------------
            // Output file per replica
            // -----------------------------

            std::string fname =
                "gcmc_T" + std::to_string(s.T) +
                "_f" + std::to_string(s.f) +
                "_rep" + std::to_string(r) + ".dat";

            std::ofstream out(fname);
            out << "# step   N   rho   U\n";

            int tune_interval = 1000;
            double target_low  = 0.3;
            double target_high = 0.7;

            // -----------------------------
            // Equilibration
            // During equilibration, adapt the displacement amplitude delta
            // based on acceptance rate of translational moves only.
            // Delta is frozen before production to preserve detailed balance.
            // -----------------------------

            for (int i = 0; i < n_equil; i++)
            {
                sim.mc_step();

                if ((i + 1) % tune_interval == 0)
                {
                    double acc =
                        (sim.move_attempts > 0)
                        ? double(sim.move_accepts) / sim.move_attempts
                        : 0.0;

                    if (acc < target_low)
                        sim.delta *= 0.9;
                    else if (acc > target_high)
                        sim.delta *= 1.1;

                    // safety bounds
                    sim.delta = std::clamp(sim.delta, 0.01, 0.5 * sim.skin);

                    // reset counters
                    sim.move_attempts = 0;
                    sim.move_accepts  = 0;
                }
            }
            std::cout << "EQUILIBRIATION DONE\n";
            std::cout << "FINAL DELTA = " << sim.delta << "\n";

            double N_sum = 0.0;
            double N2_sum = 0.0;
            double rho_sum = 0.0;
            double rho2_sum = 0.0;
            double U_sum = 0.0;
            int n_samples = 0;

            // -----------------------------
            // Production
            // -----------------------------

            for (int i = 0; i < n_prod; i++)
            {
                if (i % (n_prod / 10) == 0 && i > 0)
                {
                    std::cout << "Production progress: "
                    << (100.0 * i / n_prod) << "%\n";
                }

                sim.mc_step();

                if (i % stride == 0)
                {
                    double rho = sim.n / sim.box_volume;
                    // --- tail correction ---
                    double rc = sim.r_cut;
                    double u_tail_per_particle =
                        (8.0 * M_PI * rho / 3.0) *
                        ((1.0 / (3.0 * std::pow(rc, 9))) -
                        (1.0 / std::pow(rc, 3)));

                    double U_tail = sim.n * u_tail_per_particle;
                    double U_corr = sim.u_total + U_tail;

                    // file output (unchanged)
                    out << i << " "
                        << sim.n << " "
                        << rho << " "
                        << U_corr << "\n";

                    // accumulators
                    N_sum   += sim.n;
                    N2_sum  += sim.n * sim.n;
                    rho_sum += rho;
                    rho2_sum+= rho * rho;
                    U_sum   += U_corr;
                    n_samples++;
                }
            }
            out.close();

            // replica averages
                N_rep.push_back(N_sum / n_samples);
                rho_rep.push_back(rho_sum / n_samples);

                std::cout << "Replica " << r
                        << " : <N> = " << N_rep.back()
                        << " , <rho> = " << rho_rep.back()
                        << "\n";
        }
            // -----------------------------------------------------
            // Average over replicas
            // -----------------------------------------------------

            double N_avg = std::accumulate(N_rep.begin(), N_rep.end(), 0.0)
                        / N_rep.size();
            double rho_avg = std::accumulate(rho_rep.begin(), rho_rep.end(), 0.0)
                            / rho_rep.size();

            double N_var = 0.0, rho_var = 0.0;
            for (size_t i = 0; i < N_rep.size(); i++)
            {
                N_var   += (N_rep[i]   - N_avg)   * (N_rep[i]   - N_avg);
                rho_var += (rho_rep[i] - rho_avg) * (rho_rep[i] - rho_avg);
            }

            double N_err =
                std::sqrt(N_var / (N_rep.size() - 1)) / std::sqrt(N_rep.size());
            double rho_err =
                std::sqrt(rho_var / (rho_rep.size() - 1)) / std::sqrt(rho_rep.size());

            // -----------------------------
            // Final report for this state
            // -----------------------------

            std::cout << "----------------------------------------\n";
            std::cout << "FINAL AVERAGES (replica-averaged)\n";
            std::cout << "<N>   = " << N_avg   << " +/- " << N_err << "\n";
            std::cout << "<rho> = " << rho_avg << " +/- " << rho_err << "\n";
            std::cout << "----------------------------------------\n";
    }
    return 0;
}