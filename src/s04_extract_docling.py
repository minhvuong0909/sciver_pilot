"""Bước 3: trích bảng từ ảnh bằng Docling (E0).  Chạy trong môi trường "docling" riêng.

Docling nhận ảnh PNG/JPEG trực tiếp (layout model + TableFormer + OCR).
Mỗi ảnh -> chọn bảng lớn nhất Docling tìm được -> HTML -> grid (cùng hàm với bảng chuẩn).
Nếu Docling không tìm thấy bảng: lưu bảng rỗng và ghi lý do (KHÔNG loại mẫu).

  python src/s04_extract_docling.py --data_root data/SciVer --samples work/samples.json --out_dir work
"""
import argparse
import os
import time
import traceback

from common import PathResolver, evidence_items, load_json, save_json
from tables import make_table


def build_converter(do_ocr=True):
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter
    try:  # cấu hình rõ ràng nếu phiên bản Docling hỗ trợ
        from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
        from docling.document_converter import ImageFormatOption
        opts = PdfPipelineOptions()
        opts.do_ocr = do_ocr
        opts.do_table_structure = True
        opts.table_structure_options.mode = TableFormerMode.ACCURATE
        opts.table_structure_options.do_cell_matching = False  # ảnh không có lớp chữ PDF
        return DocumentConverter(format_options={InputFormat.IMAGE: ImageFormatOption(pipeline_options=opts)})
    except Exception:  # noqa: BLE001
        return DocumentConverter()


def docling_version():
    try:
        from importlib.metadata import version
        return version("docling")
    except Exception:  # noqa: BLE001
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--samples", default="work/samples.json")
    ap.add_argument("--out_dir", default="work")
    ap.add_argument("--no_ocr", action="store_true")
    args = ap.parse_args()

    res = PathResolver(args.data_root)
    conv = build_converter(do_ocr=not args.no_ocr)
    ver = docling_version()
    out_dir = os.path.join(args.out_dir, "extracted_docling")
    done = 0
    for q in load_json(args.samples):
        for t, item, img in evidence_items(q):
            if t != "table":
                continue
            key = f"{q['_qid']}__{item}"
            img_path = res.resolve(img)
            t0 = time.time()
            html, status = None, "ok"
            try:
                doc = conv.convert(img_path).document
                tabs = list(doc.tables)
                if not tabs:
                    status = "no_table_found"
                else:
                    htmls = [tb.export_to_html(doc=doc) for tb in tabs]
                    html = max(htmls, key=len)
                    if len(tabs) > 1:
                        status = f"ok_{len(tabs)}_tables_took_largest"
            except Exception as e:  # noqa: BLE001
                status = f"error:{type(e).__name__}:{e}"
                traceback.print_exc()
            tab = make_table(html, "docling") if html else {"grid": [], "n_header_rows": 0, "source": "docling"}
            save_json({"key": key, "status": status, "seconds": round(time.time() - t0, 2),
                       "docling_version": ver, "image_path": img, "table": tab, "raw_html": html},
                      os.path.join(out_dir, key + ".json"))
            done += 1
            print(f"{key}: {status} ({len(tab['grid'])} dòng, {time.time() - t0:.1f}s)")
    print(f"Xong {done} ảnh bảng -> {out_dir} (docling {ver})")


if __name__ == "__main__":
    main()
