/*
// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////

#include <algorithm>

#include "pass_manager.h"
#include "program_node.h"
#include "mutable_data_inst.h"
#include "concatenation_inst.h"
#include "scale_inst.h"
#include "tensor_type.h"
#include <memory>
#include <vector>
#include <stdexcept>

void compile_program::run(program_impl& p) {
   p.get_engine().compile_program(p);
}
