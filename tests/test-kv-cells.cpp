#include "../src/llama-kv-cells.h"

#include <cstdint>
#include <vector>

#define CHECK(expr) do { if (!(expr)) return __LINE__; } while (0)

static bool check_range(const llama_kv_cells & cells, llama_seq_id seq_id, uint32_t first, uint32_t last) {
    uint32_t actual_first = 0;
    uint32_t actual_last  = 0;
    const bool found = cells.seq_cells_range(seq_id, actual_first, actual_last);
    return found == (first != last) && actual_first == first && actual_last == last;
}

int main() {
    llama_kv_cells cells;
    cells.resize(16);

    cells.pos_set(5,  10);
    cells.seq_add(5, 0);

    cells.pos_set(2,  20);
    cells.seq_add(2, 0);
    cells.seq_add(2, 1);

    cells.pos_set(7,  20);
    cells.seq_add(7, 0);

    cells.pos_set(12, 30);
    cells.seq_add(12, 0);

    uint32_t cell = 0;
    CHECK(cells.seq_pos_find(0, 20, 30, cell) && cell == 2);

    // Removing sequence 0 must preserve a cell shared with sequence 1.
    CHECK(!cells.seq_rm(cell, 0));
    CHECK(cells.seq_has(2, 1));

    CHECK(cells.seq_pos_find(0, 20, 30, cell) && cell == 7);
    CHECK(cells.seq_rm(cell, 0));
    CHECK(cells.is_empty(7));
    CHECK(!cells.seq_pos_find(0, 20, 30, cell));

    uint32_t first = 0;
    uint32_t last  = 0;
    CHECK(cells.seq_cells_range(0, first, last));
    CHECK(first == 5 && last == 13);

    CHECK(cells.seq_pos_find(0, 30, 31, cell) && cell == 12);
    CHECK(cells.seq_rm(cell, 0));
    CHECK(cells.seq_cells_range(0, first, last));
    CHECK(first == 5 && last == 6);

    CHECK(cells.seq_pos_find(0, 0, 11, cell) && cell == 5);
    CHECK(cells.seq_rm(cell, 0));
    CHECK(!cells.seq_cells_range(0, first, last));

    CHECK(check_range(cells, 1, 2, 3));
    cells.reset();
    CHECK(check_range(cells, 0, 0, 0));
    CHECK(check_range(cells, 1, 0, 0));

    cells.pos_set(2, 10);
    cells.seq_add(2, 0);
    cells.seq_add(2, 1);
    cells.pos_set(12, 30);
    cells.seq_add(12, 0);

    // Position shifts must preserve physical bounds unless they remove a cell.
    CHECK(!cells.pos_add(2, 10));
    cells.pos_div(12, 2);
    CHECK(check_range(cells, 0, 2, 13));
    CHECK(check_range(cells, 1, 2, 3));
    CHECK(cells.seq_pos_find(0, 15, 16, cell) && cell == 12);
    CHECK(cells.seq_pos_find(1, 20, 21, cell) && cell == 2);
    CHECK(!cells.seq_pos_find(0, 30, 31, cell));

    CHECK(cells.pos_add(2, -21));
    CHECK(check_range(cells, 0, 12, 13));
    CHECK(check_range(cells, 1, 0, 0));
    CHECK(cells.pos_add(12, -16));
    CHECK(check_range(cells, 0, 0, 0));
    CHECK(cells.get_used() == 0);
    cells.reset_shift();

    cells.pos_set(2, 10);
    cells.seq_add(2, 0);
    cells.seq_add(2, 1);
    cells.pos_set(7, 20);
    cells.seq_add(7, 0);
    cells.pos_set(12, 30);
    cells.seq_add(12, 2);

    const auto saved = cells.cp(2, 11);
    CHECK(!cells.seq_keep(2, 1));
    CHECK(check_range(cells, 0, 7, 8));
    CHECK(check_range(cells, 1, 2, 3));
    cells.rm(7);
    CHECK(check_range(cells, 0, 0, 0));

    cells.set(2, saved);
    CHECK(check_range(cells, 0, 2, 8));
    CHECK(check_range(cells, 1, 2, 3));
    CHECK(check_range(cells, 2, 12, 13));
    CHECK(cells.seq_pos_find(0, 10, 11, cell) && cell == 2);

    const auto scattered = cells.cp(std::vector<uint32_t>{ 12, 2, 7 });
    cells.rm(2);
    cells.rm(7);
    cells.rm(12);
    cells.set(std::vector<uint32_t>{ 1, 10, 14 }, scattered);
    CHECK(check_range(cells, 0, 10, 15));
    CHECK(check_range(cells, 1, 10, 11));
    CHECK(check_range(cells, 2, 1, 2));
    CHECK(cells.seq_pos_find(0, 10, 11, cell) && cell == 10);
    CHECK(cells.seq_pos_find(0, 20, 21, cell) && cell == 14);
    CHECK(cells.get_used() == 3);

    // Restoring empty cells must remove their old sequence membership.
    cells.set(2, saved);
    CHECK(check_range(cells, 0, 2, 15));
    CHECK(check_range(cells, 1, 2, 3));
    CHECK(check_range(cells, 2, 1, 13));
    CHECK(!cells.seq_has(10, 1));
    CHECK(cells.seq_pos_find(1, 10, 11, cell) && cell == 2);
    CHECK(cells.get_used() == 5);

    cells.resize(8);
    CHECK(cells.get_used() == 0);
    CHECK(check_range(cells, 0, 0, 0));
    CHECK(check_range(cells, 1, 0, 0));
    CHECK(check_range(cells, 2, 0, 0));

    return 0;
}
