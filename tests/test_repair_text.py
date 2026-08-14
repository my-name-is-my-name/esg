from __future__ import annotations

import unittest

from esg.repair_text import append_interval_tokens, clean_repair_section, interval_tokens


class RepairTextTests(unittest.TestCase):
    def test_keeps_location_table_and_removes_figures_and_other_tables(self) -> None:
        source = """На левой консоли крыла у стрингера 7 (рисунки Рисунок 5.11, Рисунок 5.12) два крепежа смещены. Детали ремонта представлены на рисунках Рисунок 8 – Рисунок 15.

Таблица 5.11 – Данные о крепеже

| Крепеж | Материал |
| --- | --- |
| ASNA2027VHK4 | Титан |

Таблица 5.12 – Описание области

| Р.з. | Область ремонта | Тип отклонения |
| --- | --- | --- |
| 1 | Левая консоль крыла, стрингер 7, нервюра 19, направляющая 4 закрылка | смещение крепежа |

Рисунок 5.11 – Описание ремонта

Стрингер 7Стрингер 8Зона ремонтаНаправляющая 4 закрылка
"""

        result = clean_repair_section(source)

        self.assertIn("На левой консоли крыла у стрингера 7", result)
        self.assertIn("Область ремонта: Левая консоль крыла, стрингер 7, нервюра 19", result)
        self.assertNotIn("ASNA2027VHK4", result)
        self.assertNotIn("Рисунок 5.11", result)
        self.assertNotIn("Рисунок 15", result)
        self.assertNotIn("Стрингер 7Стрингер 8", result)

    def test_interval_tokens_expand_structural_ranges(self) -> None:
        tokens = interval_tokens("обшивка крыла между нервюрами 1-3 и стрингерами 6 – 7")

        self.assertIn("inv_rib_0001", tokens)
        self.assertIn("inv_rib_0003", tokens)
        self.assertIn("inv_stringer_0006", tokens)
        self.assertIn("inv_stringer_0007", tokens)

    def test_interval_tokens_are_prepended_for_retrieval(self) -> None:
        result = append_interval_tokens("у шпангоута 8 между стрингерами 6-7")

        self.assertTrue(result.startswith("inv_frame_0008 inv_stringer_0006 inv_stringer_0007\n"))


if __name__ == "__main__":
    unittest.main()
