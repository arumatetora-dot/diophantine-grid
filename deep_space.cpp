#include <iostream>
#include <vector>
#include <chrono>
#include <string>
#include <omp.h>
#include <gmp.h>
#include <gmpxx.h>

#define K_TARGET 114

// 128-bit整数をパースする関数
__uint128_t parse_uint128(const std::string& str) {
    __uint128_t res = 0;
    for (char c : str) res = res * 10 + (c - '0');
    return res;
}

// 128-bitからmpz_classへの安全な変換
mpz_class to_mpz(__uint128_t val) {
    uint64_t lo = (uint64_t)val;
    uint64_t hi = (uint64_t)(val >> 64);
    mpz_class res = hi;
    res <<= 64;
    res += lo;
    return res;
}

// 高速モジュラー冪乗
inline uint64_t mod_pow(uint64_t base, uint64_t exp, uint64_t mod) {
    uint64_t res = 1;
    base %= mod;
    while (exp > 0) {
        if (exp % 2 == 1) res = (res * base) % mod;
        base = (base * base) % mod;
        exp /= 2;
    }
    return res;
}

// 128bit y に対するオイラーの平方剰余フィルター (64bitダウンキャスト高速判定)
inline bool is_qr(__uint128_t y, uint64_t u, uint64_t P) {
    uint64_t y_mod = (uint64_t)(y % P);
    uint64_t u_mod = u % P;
    
    uint64_t y3 = (y_mod * y_mod) % P;
    y3 = (y3 * y_mod) % P;
    
    uint64_t u3 = (u_mod * u_mod) % P;
    u3 = (u3 * u_mod) % P;
    
    long long V = (4LL * y3 - 456LL - u3) % (long long)P;
    if (V < 0) V += P;
    
    uint64_t inv3u = mod_pow((3 * u_mod) % P, P - 2, P);
    uint64_t S = (V * inv3u) % P;
    
    if (S == 0) return true;
    return mod_pow(S, (P - 1) / 2, P) == 1;
}

int main(int argc, char** argv) {
    if (argc < 3) return 1;
    __uint128_t y_start = parse_uint128(argv[1]);
    __uint128_t y_end = parse_uint128(argv[2]);
    
    uint64_t u_min = 1;
    uint64_t u_max = 50000;

    std::cout << "====================================================\n";
    std::cout << " 🌌 [CPU Deep Space] 128-bit Native Engine\n";
    std::cout << " > y_start : " << argv[1] << "\n";
    std::cout << " > y_end   : " << argv[2] << "\n";
    std::cout << "====================================================\n";

    auto start_time = std::chrono::high_resolution_clock::now();
    uint64_t total_skipped = 0;
    bool found = false;

    // u を並列化し、各スレッドが指定された y_start 〜 y_end の区間を処理
    #pragma omp parallel for schedule(dynamic, 100) reduction(+:total_skipped)
    for (uint64_t u = u_min; u <= u_max; ++u) {
        if (found) continue;

        // モジュラー根の計算
        std::vector<uint64_t> roots;
        uint64_t mod = 3 * u;
        for (uint64_t r = 0; r < mod; r++) {
            uint64_t r_mod = r % mod;
            uint64_t r2 = (r_mod * r_mod) % mod;
            uint64_t r3 = (r2 * r_mod) % mod;
            
            uint64_t u_mod = u % mod;
            uint64_t u2 = (u_mod * u_mod) % mod;
            uint64_t u3 = (u2 * u_mod) % mod;
            
            uint64_t term1 = (4 * r3) % mod;
            uint64_t term2 = (456 + u3) % mod;
            if (term1 == term2) roots.push_back(r);
        }

        if (roots.empty()) continue;

        for (uint64_t r : roots) {
            if (found) break;
            
            // y_start を超える最初の k を算出
            __uint128_t k_start = 0;
            if (y_start > r) {
                k_start = (y_start - r + mod - 1) / mod;
            }
            
            for (__uint128_t k = k_start; ; ++k) {
                __uint128_t y = k * mod + r;
                if (y > y_end) break;
                total_skipped++;

                // 4連オイラーフィルター (非平方数を 93% 枝刈り)
                if (!is_qr(y, u, 1000000007ULL)) continue;
                if (!is_qr(y, u, 1000000009ULL)) continue;
                if (!is_qr(y, u, 1000000021ULL)) continue;
                if (!is_qr(y, u, 1000000033ULL)) continue;

                // フィルター生存者のみ GMP (mpz_class) へエスカレーション
                mpz_class my = to_mpz(y);
                mpz_class mu = u;
                mpz_class y3 = my * my * my;
                mpz_class u3 = mu * mu * mu;
                mpz_class V = 4 * y3 - 456 - u3;
                mpz_class S = V / (3 * mu);

                if (mpz_perfect_square_p(S.get_mpz_t())) {
                    mpz_class root;
                    mpz_sqrt(root.get_mpz_t(), S.get_mpz_t());
                    
                    if (root % 2 == mu % 2) {
                        mpz_class two_x = (root > mu) ? (root - mu) : (mu - root);
                        if (two_x > 0 && two_x % 2 == 0) {
                            mpz_class x = two_x / 2;
                            mpz_class z = x + mu;

                            #pragma omp critical
                            {
                                std::cout << "\n**************************************************\n";
                                std::cout << "🚨 [!!! K=114 DEEP SPACE 完全解発見 !!!]\n";
                                std::cout << "  x = " << x.get_str() << "\n";
                                std::cout << "  y = " << my.get_str() << "\n";
                                std::cout << "  z = -" << z.get_str() << "\n";
                                std::cout << "  (u = " << u << ")\n";
                                std::cout << "**************************************************\n";
                                found = true;
                            }
                        }
                    }
                }
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;
    std::cout << " | 検証 y 軌道: " << total_skipped << " 点\n";
    std::cout << " | 完了時間   : " << diff.count() << " 秒\n";

    return 0;
}
