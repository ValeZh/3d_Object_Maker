from src.generator.ai.ollama_client import extract_attributes_with_ollama

@app.post("/api/generate-from-text")
async def generate_from_text(request: Request):
    payload = await request.json()
    text = payload.get("text", "").strip()
    if not text:
        return JSONResponse({"error": "Пустой текст"}, status_code=400)

    # ← Вот здесь используем Ollama
    attrs = extract_attributes_with_ollama(text)

    shape = attrs.get("shape") or "sphere"
    texture = attrs.get("texture") or "metallic"
    color = attrs.get("color") or "#ff00ff"
    color_rgb = hex_to_rgb(color) if isinstance(color, str) and color.startswith("#") else [1, 0, 1]

    logger.info(f"Генерация из текста: '{text}' → {shape=}, {texture=}, {color=}, additional={attrs.get('additional')}")

    result = create_gan_object(
        shape=shape, color=color_rgb, texture=texture,
        output_dir=OUTPUT_DIR, textures_dir=TEXTURES_DIR
    )

    obj_path = Path(result["obj_path"])
    mtl_path = Path(result["mtl_path"])

    tex_paths = []
    for ext in ["jpg", "jpeg", "png"]:
        p = obj_path.parent / f"{texture}.{ext}"
        if p.exists():
            tex_paths.append(p)

    zip_name = f"text_{obj_path.stem}.zip"
    zip_path = obj_path.parent / zip_name
    make_zip(obj_path, mtl_path, tex_paths, zip_path)

    rel_path = zip_path.relative_to(OUTPUT_DIR).as_posix()
    return {
        "zip_url": f"/files/{rel_path}",
        "attributes": attrs  # ← отдаём фронтенду сразу
    }