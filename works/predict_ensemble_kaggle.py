def main(images_dir, models_dir, config_path, out):
    if out is not None and not out.endswith((".json")):
        raise ValueError("The output file must be a JSON file.")

    # create list of test images in coco format since mmdet inference expects this
    annotations, image_id_to_filename, image_dict = create_empty_coco_annotations(
        images_dir
    )

    with open("annotations.json", "w") as f:
        json.dump(annotations, f)

    checkpoint_paths = list(Path(models_dir).glob("*.pth"))

    # list of predictions from each model
    result_files = []

    ################################################## INFERENCE ##################################################
    for checkpoint_path in checkpoint_paths:
        cfg = Config.fromfile(config_path)
        cfg.data.test.ann_file = "annotations.json"
        cfg.data.test.img_prefix = images_dir
        cfg = compat_cfg(cfg)

        setup_multi_processes(cfg)
        cfg.gpu_ids = [
            0,
        ]
        cfg.device = get_device()

        test_dataloader_default_args = dict(
            samples_per_gpu=1, workers_per_gpu=2, dist=False, shuffle=False
        )

        cfg.data.test.test_mode = True
        if cfg.data.test_dataloader.get("samples_per_gpu", 1) > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)

        test_loader_cfg = {
            **test_dataloader_default_args,
            **cfg.data.get("test_dataloader", {}),
        }

        # build the dataloader
        dataset = build_dataset(cfg.data.test)
        data_loader = build_dataloader(dataset, **test_loader_cfg)

        # build the model and load checkpoint
        cfg.model.train_cfg = None
        model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
        fp16_cfg = cfg.get("fp16", None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        checkpoint = load_checkpoint(model, str(checkpoint_path), map_location="cpu")
        model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)
        model_name = checkpoint_path.name.split(".")[0]

        # run inference with the model on data
        outputs = single_gpu_test(model, data_loader, False, None, 0.3)

        if not os.path.exists("./predictions"):
            os.mkdir("./predictions")

        dataset.format_results(outputs, jsonfile_prefix=f"./predictions/{model_name}")

        raw_bbox_predictions = json.load(
            open(f"./predictions/{model_name}.bbox.json", "r")
        )

        # prune and format the raw predictions for fusion
        annotations_ = []

        for r in raw_bbox_predictions:
            if r["score"] > 0.1:
                anno = {}
                anno["id"] = r["image_id"]
                anno["bbox"] = r["bbox"]
                anno["category_id"] = r["category_id"]
                anno["file_name"] = image_id_to_filename[str(r["image_id"])]
                anno["score"] = r["score"]
                annotations_.append(anno)

        annotations = dict(annotations=annotations_)
        json.dump(
            annotations, open(f"./predictions/{model_name}_threshold=0.1.json", "w")
        )

        result_files.append(
            {
                "name": model_name,
                "path": f"./predictions/{model_name}_threshold=0.1.json",
                "weight": 0.2,  # 2022-05-20
            }
        )

    ################################################## ENSEMBLE ##################################################
    # format result files for fusion
    format_annotation_dict = {}
    file_paths = []
    for rf in result_files:
        format_ann = format_annotations(rf["path"], image_dict)

        format_annotation_dict[rf["name"]] = format_ann

        file_paths.extend(format_ann.keys())

    uniq_file_paths = sorted(list(set(file_paths)))

    weight_dict = {r["name"]: r["weight"] for r in result_files}

    fusions_dict = process_fusion(
        uniq_file_paths[:], NMS_THRESH, BOX_THRESH, weight_dict, format_annotation_dict
    )

    # convert fusion results to submission format
    items = []
    cnt_id = 0
    cnt_lower_threshold = 0
    threshold = PP_THRESH

    for file_name, info in fusions_dict.items():
        # print(file_name)

        image_info = image_dict[file_name]

        score_indexes = np.argsort(info["scores"])

        if len(info["scores"]) == 0:
            continue

        # filter max score
        index = np.argmax(info["scores"])

        cnt_annotate = 0
        found_category_id = None
        for index in score_indexes:

            label = info["labels"][index]
            score = info["scores"][index]
            box = info["boxes"][index]

            if score < threshold:
                continue

            category_id = int(label)
            coco_bbox = convert_norm_box_to_coco_bbox(
                box, image_info["width"], image_info["height"]
            )

            item = {
                "id": cnt_id,
                "bbox": coco_bbox,
                "category_id": category_id,
                "file_name": file_name,
                "score": float(score),
            }
            items.append(item)

            cnt_id += 1
            cnt_annotate += 1
            found_category_id = category_id

    missing_pred_images = set(image_dict.keys()) - set(
        list(map(lambda r: r["file_name"], items))
    )

    # fill missing predictions with the selected model's outputs
    selected_model = "tood_r101_dconv_10epoch"
    missing_items = []
    for missing_im in missing_pred_images:
        file_name = missing_im
        image_info = image_dict[file_name]

        info = format_annotation_dict[selected_model].get(file_name, None)
        if info is None:
            print("Not found: {}".format(file_name))
            continue

        # filter max score
        index = np.argmax(info["scores_list"])

        label = info["labels_list"][index]
        score = info["scores_list"][index]
        box = info["boxes_list"][index]
        category_id = int(label)

        coco_bbox = convert_norm_box_to_coco_bbox(
            box, image_info["width"], image_info["height"]
        )

        item = {
            "id": cnt_id,
            "bbox": coco_bbox,
            "category_id": category_id,
            "file_name": missing_im,
            "score": float(score),
        }
        items.append(item)
        cnt_id += 1

    submission_output = {"annotations": items}
    json.dump(submission_output, open(out, "w"), indent=2, sort_keys=False)