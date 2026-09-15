#include "../src/llama-kv-cells.h"

#include <cstdint>

#define CHECK(expr) do { if (!(expr)) return __LINE__; } while (0)

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

    return 0;
}
