# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

#!/bin/bash

# Check if ProteinMPNN directory exists
if [ ! -d "ProteinMPNN" ]; then
    echo "Error: ProteinMPNN directory not found. Please run this script from the proteina root directory."
    exit 1
fi

cd ProteinMPNN

# Function to check if weights already exist
check_weights_exist() {
    local weight_dir=$1
    shift  # Remove first argument, rest are file names
    
    if [ -d "$weight_dir" ]; then
        local all_exist=true
        for file in "$@"; do
            if [ ! -f "$weight_dir/$file" ]; then
                all_exist=false
                break
            fi
        done
        if [ "$all_exist" = true ]; then
            return 0  # All files exist
        fi
    fi
    return 1  # Files missing
}

# Check CA model weights
CA_WEIGHTS=("v_48_002.pt" "v_48_010.pt" "v_48_020.pt")
if check_weights_exist "ca_model_weights" "${CA_WEIGHTS[@]}"; then
    echo "CA model weights already exist. Skipping download."
else
    echo "Downloading CA model weights..."
    rm -rf ca_model_weights
    mkdir -p ca_model_weights
    cd ca_model_weights
    wget -q https://github.com/dauparas/ProteinMPNN/raw/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57/ca_model_weights/v_48_002.pt
    wget -q https://github.com/dauparas/ProteinMPNN/raw/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57/ca_model_weights/v_48_010.pt
    wget -q https://github.com/dauparas/ProteinMPNN/raw/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57/ca_model_weights/v_48_020.pt
    cd ..
    echo "CA model weights downloaded successfully."
fi

# Check vanilla model weights
VANILLA_WEIGHTS=("v_48_002.pt" "v_48_010.pt" "v_48_020.pt" "v_48_030.pt")
if check_weights_exist "vanilla_model_weights" "${VANILLA_WEIGHTS[@]}"; then
    echo "Vanilla model weights already exist. Skipping download."
else
    echo "Downloading vanilla model weights..."
    rm -rf vanilla_model_weights
    mkdir -p vanilla_model_weights
    cd vanilla_model_weights
    wget -q https://github.com/dauparas/ProteinMPNN/raw/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57/vanilla_model_weights/v_48_002.pt
    wget -q https://github.com/dauparas/ProteinMPNN/raw/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57/vanilla_model_weights/v_48_010.pt
    wget -q https://github.com/dauparas/ProteinMPNN/raw/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57/vanilla_model_weights/v_48_020.pt
    wget -q https://github.com/dauparas/ProteinMPNN/raw/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57/vanilla_model_weights/v_48_030.pt
    cd ..
    echo "Vanilla model weights downloaded successfully."
fi

cd ..
echo "ProteinMPNN weights setup complete!"