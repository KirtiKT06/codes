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
#include <iomanip>
#include <sstream>
#include <numeric> // for std::iota
#include <string>  // forstd::string

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
    double P_avg;      // average pressure
    double U_avg;      // average energy per particle
    double beta_mu;    // Widom excess chemical potential
    int widom_samples; // total number of Widom insertions
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
    int N;          // number of particles
    double box_len; // simulation box length
    double temp;    // reduced temprature
    double beta;    // inverse temperature (1/T)
    double u_total; // total potential energy
    double rho;     // reduced density
    double delta;   // MC displacement amplitude

    double r_cut = 2.5; // LJ cutoff radius
    double r_cut_sq;    // squared cutoff radius

    // -------------------------
    // Neighbor list parameters
    // -------------------------
    std::vector<std::vector<int>> neigh; // verlet neighbour list
    double r_list;                       // neighbour list cutoff
    double r_list_sq;                    // squared neighbour list cutoff
    double skin;                         // skin distance

    // -------------------------
    // Cell list parameters
    // -------------------------
    int n_cells;
    double cell_size;
    std::vector<std::vector<int>> cells;

    // -------------------------
    // Particle coordinates
    // -------------------------
    std::vector<double> x, y, z;
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
        beta = 1.0 / temp;
        box_len = cbrt(N / rho);

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
        double a = box_len / n_cell;

        u_total = 0.0;

        int i = 0;
        for (int ix = 0; ix < n_cell; ix++)
        {
            for (int iy = 0; iy < n_cell; iy++)
            {
                for (int iz = 0; iz < n_cell; iz++)
                {
                    if (i < N)
                    {
                        x[i] = (ix + 0.5) * a;
                        y[i] = (iy + 0.5) * a;
                        z[i] = (iz + 0.5) * a;
                        i++;
                    }
                }
            }
        }

        // Compute initial total energy by direct summation
        u_total = 0.0;
        for (int i = 0; i < N - 1; i++)
            for (int j = i + 1; j < N; j++)
                u_total += pair_energy(i, j);
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

                double r2 = dx * dx + dy * dy + dz * dz;
                if (r2 < r_list_sq)
                {
                    neigh[i].push_back(j);
                    neigh[j].push_back(i);
                }
            }
        }
    }

    // ========================================================
    // Build cell list
    // ========================================================
    void build_cells()
    {
        cell_size = r_list;
        n_cells = std::max(1, int(box_len / cell_size));

        if (n_cells < 1)
            n_cells = 1;

        cells.assign(n_cells * n_cells * n_cells, {});

        for (int i = 0; i < N; i++)
        {
            int cx = int(x[i] / cell_size);
            int cy = int(y[i] / cell_size);
            int cz = int(z[i] / cell_size);

            cx = (cx + n_cells) % n_cells;
            cy = (cy + n_cells) % n_cells;
            cz = (cz + n_cells) % n_cells;

            int index = cx + n_cells * (cy + n_cells * cz);
            cells[index].push_back(i);
        }
    }
    template <typename Func>
    void for_each_neighbor(int i, Func func)
    {
        int cx = int(x[i] / cell_size);
        int cy = int(y[i] / cell_size);
        int cz = int(z[i] / cell_size);

        for (int dx_cell = -1; dx_cell <= 1; dx_cell++)
            for (int dy_cell = -1; dy_cell <= 1; dy_cell++)
                for (int dz_cell = -1; dz_cell <= 1; dz_cell++)
                {
                    int nx = (cx + dx_cell + n_cells) % n_cells;
                    int ny = (cy + dy_cell + n_cells) % n_cells;
                    int nz = (cz + dz_cell + n_cells) % n_cells;

                    int index = nx + n_cells * (ny + n_cells * nz);

                    for (int j : cells[index])
                    {
                        if (j == i)
                            continue;
                        func(j);
                    }
                }
    }

    // ========================================================
    // Check whether neighbor list needs rebuilding
    // ========================================================
    bool need_rebuild() const
    {
        double max_disp_sq = 0.0;
        for (int i = 0; i < N; i++)
        {
            double d_sq = dx_acc[i] * dx_acc[i] + dy_acc[i] * dy_acc[i] + dz_acc[i] * dz_acc[i];
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
        if (r2 > r_cut_sq)
            return 0.0;

        if (r2 < 1e-12)
            return 1e6;

        double inv_r2 = 1.0 / r2;
        double sr6 = inv_r2 * inv_r2 * inv_r2;
        double sr12 = sr6 * sr6;

        double u = 4.0 * (sr12 - sr6);
        
        return u;
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
        double dx_move = delta * (uni01(rng) - 0.5);
        double dy_move = delta * (uni01(rng) - 0.5);
        double dz_move = delta * (uni01(rng) - 0.5);

        // Energy before move
        double delta_u = 0.0;

        for_each_neighbor(i, [&](int j)
        {
            if (j == i) return;

            double dx = x[i] - x[j];
            dx -= box_len * std::round(dx / box_len);
            double dy = y[i] - y[j];
            dy -= box_len * std::round(dy / box_len);
            double dz = z[i] - z[j];
            dz -= box_len * std::round(dz / box_len);

            double r2 = dx*dx + dy*dy + dz*dz;

            if (r2 < r_cut_sq && r2 > 1e-12)
            {
                double inv_r2 = 1.0 / r2;
                double sr6 = inv_r2 * inv_r2 * inv_r2;
                double sr12 = sr6 * sr6;
                delta_u -= 4.0 * (sr12 - sr6);
            } 
        });

        // Apply displacement with periodic boundaries
        x[i] += dx_move;
        x[i] -= box_len * (double)std::floor(x[i] / box_len);
        y[i] += dy_move;
        y[i] -= box_len * (double)std::floor(y[i] / box_len);
        z[i] += dz_move;
        z[i] -= box_len * (double)std::floor(z[i] / box_len);

        // Energy after move
        for_each_neighbor(i, [&](int j)
        {
            if (j == i) return;

            double dx = x[i] - x[j];
            dx -= box_len * std::round(dx / box_len);
            double dy = y[i] - y[j];
            dy -= box_len * std::round(dy / box_len);
            double dz = z[i] - z[j];
            dz -= box_len * std::round(dz / box_len);

            double r2 = dx*dx + dy*dy + dz*dz;

            if (r2 < r_cut_sq && r2 > 1e-12)
            {
                double inv_r2 = 1.0 / r2;
                double sr6 = inv_r2 * inv_r2 * inv_r2;
                double sr12 = sr6 * sr6;
                delta_u += 4.0 * (sr12 - sr6);
            }
        });

        // Metropolis acceptance
        if (delta_u <= 0.0 || uni01(rng) < std::exp(-beta * delta_u))
        {
            u_total += delta_u;

            dx_acc[i] += dx_move;
            dy_acc[i] += dy_move;
            dz_acc[i] += dz_move;
            
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
        build_cells();
        reset_displacements();

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
    void compute_virial_only(double &w)
    {
        w = 0.0;

        for (int i = 0; i < N; i++)
        {
            for_each_neighbor(i, [&](int j)
            {
                if (j<=i) return;

                double dx = x[i] - x[j];
                dx -= box_len * std::round(dx / box_len);
                double dy = y[i] - y[j];
                dy -= box_len * std::round(dy / box_len);
                double dz = z[i] - z[j];
                dz -= box_len * std::round(dz / box_len);

                double r2 = dx*dx + dy*dy + dz*dz;

                if (r2 < r_cut_sq)
                {
                    double inv_r2 = 1.0 / r2;
                    double sr6 = inv_r2 * inv_r2 * inv_r2;
                    double sr12 = sr6 * sr6;

                    w += 24.0 * (2.0 * sr12 - sr6);
                }
            });
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

            double r2 = dx * dx + dy * dy + dz * dz;

            // ---- HARD-CORE PROTECTION ----
            if (r2 < 1e-12) continue;
            
            if (r2 < r_cut_sq)
            {
                // ---- CAP EXTREME OVERLAPS ----
                if (r2 < 1e-6)
                {
                    return 1e6;
                }
                double inv_r2 = 1.0 / r2;
                double sr6 = inv_r2 * inv_r2 * inv_r2;
                double sr12 = sr6 * sr6;
                double u = 4.0 * (sr12 - sr6);
                deltaU += u;
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
        bool do_widom,
        int seed)
    {
        auto t0 = std::chrono::high_resolution_clock::now();

        // -----------------------------
        // Basic thermodynamic setup
        // -----------------------------
        N = N_in;
        temp = T_in;
        rho = rho_in;
        beta = 1.0 / temp;

        indices.resize(N);
        std::iota(indices.begin(), indices.end(), 0);

        delta = 0.1;
        r_cut = 2.5;
        r_cut_sq = r_cut * r_cut;

        r_list = r_cut + 0.8;
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

        // Tail correctibuild_neighbor_list()
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
        build_cells();

        // -----------------------------
        // Equilibration
        // -----------------------------
        int n_equil = 50000;
        double acc_rate = 0.0;
        double acc_window = 0.0;
        double acc_sum = 0.0;

        for (int i = 0; i < n_equil; i++)
        {
            acc_rate = mc_cycle();
            acc_window += acc_rate;
            acc_sum += acc_rate;

            if ((i + 1) % 20 == 0)
            {
                double acc_avg = acc_window / 20.0;

                if (acc_avg > 0.5)
                    delta *= 1.1;
                else
                    delta *= 0.90;

                delta = std::clamp(delta, 0.01, 0.4 * skin);

                acc_window = 0.0; // reset window
            }
        }
        
        double acc_avg_total = acc_sum / n_equil;

        std::cout << "Equilibration done.\n";
        std::cout << "Final delta = " << delta << "\n";
        std::cout << "Acceptance rate = " << acc_avg_total << "\n";

        // -----------------------------
        // Production
        // -----------------------------
        int n_production = 600000;
        int sample_stride = 100;
        int widom_stride = 2 * sample_stride;
        int insertion_per_config = 2000;

        double u_sum = 0.0, p_sum = 0.0;
        int n_samples = 0;

        double widom_sum = 0.0;
        int widom_count = 0;

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

                if (!std::isfinite(u_total))
                {
                    std::cerr << "Energy blew up!\n";
                    exit(1);
                }
                double p_inst = rho * temp + w / (3.0 * V);
                p_inst += p_tail_const;

                double u_inst = u_total + N * u_tail_per_particle;

                u_sum += u_inst;
                p_sum += p_inst;
                n_samples++;
            }

            // ---- Widom insertion
            if (do_widom && i % widom_stride == 0)
            {
                for (int k = 0; k < insertion_per_config; k++)
                {
                    double dU = widom_insertion_energy();
                    widom_sum += std::exp(-beta * dU);
                    widom_count++;
                }
            }
        }

        std::cout << "Widom diagnostics:\n";
        std::cout << "Total Widom insertions = " << widom_count << "\n";
        std::cout << "<exp(-beta ΔU)> = "
                  << (widom_sum / widom_count) << "\n";

        // -----------------------------
        // Final averages
        // -----------------------------
        mc_results result;
        result.P_avg = p_sum / n_samples;
        result.U_avg = (u_sum / n_samples) / N;
        result.beta_mu =
            do_widom
                ? -std::log(widom_sum / widom_count) + beta * mu_tail_widom
                : NAN;
        result.widom_samples = widom_count;

        auto t1 = std::chrono::high_resolution_clock::now();
        std::cout << "Runtime (s) = "
                  << std::chrono::duration<double>(t1 - t0).count()
                  << std::endl;

        return result;
    }
};

int main(int argc, char *argv[])
{
    mc_nvt sim;

    std::cout << "STARTING SIMULATION\n";

    double T = 1.12;
    int n_replica = 25;

    std::vector<double> rhos;

    for (double rho = 0.02; rho <= 0.20; rho += 0.01)
        rhos.push_back(rho);

    // Loop over densities
    for (double rho : rhos)
    {
        // -------- Create clean filename --------
        std::ostringstream fname_stream;
        fname_stream << "exam_T"
                     << std::fixed << std::setprecision(2) << T
                     << "_rho"
                     << std::fixed << std::setprecision(2) << rho
                     << ".dat";

        std::string fname = fname_stream.str();

        std::ofstream eos_out(fname);

        if (!eos_out.is_open())
        {
            std::cerr << "Error opening file: " << fname << "\n";
            return 1;
        }

        // -------- File header --------
        eos_out << "# T rho P Z beta_mu_total U_per_particle\n";

        std::cout << "\n====================================\n";
        std::cout << "Running rho = " << rho << "\n";
        std::cout << "Output file: " << fname << "\n";
        std::cout << "====================================\n";
        eos_out << std::setprecision(10);
        // -------- Replica loop --------
        for (int k = 0; k < n_replica; k++)
        {
            int seed = 1234 + 100 * k;

            auto res = sim.run_single_state(
                T, rho, 256,
                true, // Widom ON
                seed);
            double rhoT = rho * T;
            double Z = res.P_avg / rhoT;
            double beta_mu_total = std::log(rho) + res.beta_mu;

            // ---- Write to file ----
            eos_out << T << " "
                    << rho << " "
                    << res.P_avg << " "
                    << Z << " "
                    << beta_mu_total << " "
                    << res.U_avg << "\n";

            // ---- Clean console output ----
            std::cout << "Replica " << k + 1 << "/" << n_replica
                      << " | P = " << res.P_avg
                      << " | Z = " << Z
                      << " | beta_mu = " << beta_mu_total
                      << " | U_avg = " << res.U_avg
                      << "\n";
        }

        eos_out.close();
    }
    std::cout << "\nSIMULATION COMPLETE\n";
    return 0;
}