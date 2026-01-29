# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import cv2
import time
import torch

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from nanoowl.owl_predictor import (
    OwlPredictor
)
from nanoowl.owl_drawing import (
    draw_owl_output
)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="../assets/owl_glove_small.jpg")
    parser.add_argument("--prompt", type=str, default="[an owl, a glove]")
    parser.add_argument("--threshold", type=str, default="0.1,0.1")
    parser.add_argument("--output", type=str, default="../data/owl_predict_out.jpg")
    parser.add_argument("--model", type=str, default="google/owlvit-base-patch32")
    parser.add_argument("--model_path", type=str, default="google/owlvit-base-patch32")
    parser.add_argument("--image_encoder_engine", type=str, default="../data/owl_image_encoder_patch32.engine")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--num_profiling_runs", type=int, default=30)
    args = parser.parse_args()

    prompt = args.prompt.strip("][()")
    text = prompt.split(',')
    print(text)

    thresholds = args.threshold.strip("][()")
    thresholds = thresholds.split(',')
    if len(thresholds) == 1:
        thresholds = float(thresholds[0])
    else:
        thresholds = [float(x) for x in thresholds]
    print(thresholds)
    

    predictor = OwlPredictor(
        args.model,
        args.model_path,
        image_encoder_engine=args.image_encoder_engine,
        device="cuda:0"
    )

    image = cv2.imread(args.image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_size = (predictor.image_size, predictor.image_size)
    image_resized = cv2.resize(image, input_size)
    
    text_encodings = predictor.encode_text(text)
    output = predictor.predict(
        image=image_resized, 
        text=text, 
        text_encodings=text_encodings,
        threshold=thresholds,
        pad_square=False
    )

    if args.profile:
        torch.cuda.current_stream().synchronize()
        t0 = time.perf_counter_ns()
        for i in range(args.num_profiling_runs):
            output = predictor.predict(
                image=image_resized, 
                text=text, 
                text_encodings=text_encodings,
                threshold=thresholds,
                pad_square=False
            )
        torch.cuda.current_stream().synchronize()
        t1 = time.perf_counter_ns()
        dt = (t1 - t0) / 1e9
        print(f"PROFILING TIME per run: {dt/args.num_profiling_runs} seconds")
        print(f"PROFILING FPS: {args.num_profiling_runs/dt}")

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = draw_owl_output(image, input_size, output, text=text, draw_text=True)

    cv2.imwrite(args.output, image)
