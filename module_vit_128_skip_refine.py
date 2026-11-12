"""Stage II residual refinement conditioned on the coarse map and original inputs."""
import torch
import torch.nn as nn


class ConvBNAct(nn.Module):

    def __init__(self, in_ch, out_ch, k=3, s=1, p=None):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(
            in_ch, out_ch,
            kernel_size=k,
            stride=s,
            padding=p,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Conv3Block(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNAct(in_ch, out_ch, k=3, s=1),
            ConvBNAct(out_ch, out_ch, k=3, s=1),
            ConvBNAct(out_ch, out_ch, k=3, s=1),
        )

    def forward(self, x):
        return self.net(x)


class UpBlock3(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_ch, out_ch,
            kernel_size=2,
            stride=2,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.refine = Conv3Block(out_ch, out_ch)

    def forward(self, x):
        x = self.act(self.bn(self.up(x)))
        x = self.refine(x)
        return x


class SkipFuse(nn.Module):

    def __init__(self, dec_ch, skip_ch, out_ch):
        super().__init__()
        self.fuse = Conv3Block(dec_ch + skip_ch, out_ch)

    def forward(self, dec_feat, skip_feat):
        x = torch.cat([dec_feat, skip_feat], dim=1)
        x = self.fuse(x)
        return x


class RM_BM_SkipFusion(nn.Module):

    def __init__(self, rm_bm_in_ch, out_ch):
        super().__init__()
        self.fuse = Conv3Block(rm_bm_in_ch, out_ch)

    def forward(self, rm_skip, bm_skip):
        x = torch.cat([rm_skip, bm_skip], dim=1)
        x = self.fuse(x)
        return x


class PatchEncoderPool5_Half_MS(nn.Module):

    def __init__(self, in_ch=1):
        super().__init__()
        self.b1 = Conv3Block(in_ch, 16)
        self.p1 = nn.MaxPool2d(2)

        self.b2 = Conv3Block(16, 24)
        self.p2 = nn.MaxPool2d(2)

        self.b3 = Conv3Block(24, 32)
        self.p3 = nn.MaxPool2d(2)

        self.b4 = Conv3Block(32, 48)
        self.p4 = nn.MaxPool2d(2)

        self.b5 = Conv3Block(48, 80)
        self.p5 = nn.MaxPool2d(2)

        self.to128 = nn.Conv2d(80, 128, kernel_size=1, bias=False)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        s1 = self.b1(x)
        x = self.p1(s1)

        s2 = self.b2(x)
        x = self.p2(s2)

        s3 = self.b3(x)
        x = self.p3(s3)

        s4 = self.b4(x)
        x = self.p4(s4)

        s5 = self.b5(x)
        x = self.p5(s5)

        s6 = self.to128(x)
        tok = self.pool(s6).flatten(1)

        feats = {
            "s1": s1,
            "s2": s2,
            "s3": s3,
            "s4": s4,
            "s5": s5,
            "s6": s6,
        }
        return tok, feats


class GlobalDecoder128_From4_Skip6(nn.Module):

    def __init__(self):
        super().__init__()

        self.stem = UpBlock3(128, 128)
        self.up1 = UpBlock3(128, 80)
        self.up2 = UpBlock3(80, 48)
        self.up3 = UpBlock3(48, 32)
        self.up4 = UpBlock3(32, 24)
        self.up5 = UpBlock3(24, 16)

        self.fuse8 = SkipFuse(dec_ch=128, skip_ch=128, out_ch=128)
        self.fuse16 = SkipFuse(dec_ch=80, skip_ch=80, out_ch=80)
        self.fuse32 = SkipFuse(dec_ch=48, skip_ch=48, out_ch=48)
        self.fuse64 = SkipFuse(dec_ch=32, skip_ch=32, out_ch=32)
        self.fuse128 = SkipFuse(dec_ch=24, skip_ch=24, out_ch=24)
        self.fuse256 = SkipFuse(dec_ch=16, skip_ch=16, out_ch=16)

        self.out = nn.Conv2d(16, 1, kernel_size=1)

    def forward(
        self,
        x,
        skip8=None,
        skip16=None,
        skip32=None,
        skip64=None,
        skip128=None,
        skip256=None,
    ):
        x = self.stem(x)
        if skip8 is not None:
            x = self.fuse8(x, skip8)

        x = self.up1(x)
        if skip16 is not None:
            x = self.fuse16(x, skip16)

        x = self.up2(x)
        if skip32 is not None:
            x = self.fuse32(x, skip32)

        x = self.up3(x)
        if skip64 is not None:
            x = self.fuse64(x, skip64)

        x = self.up4(x)
        if skip128 is not None:
            x = self.fuse128(x, skip128)

        x = self.up5(x)
        if skip256 is not None:
            x = self.fuse256(x, skip256)

        x = self.out(x)
        return x


class DecodeConcatFusion_Half(nn.Module):
    def __init__(self, in_ch=256, out_ch=128):
        super().__init__()
        self.fuse = Conv3Block(in_ch, out_ch)

    def forward(self, rm_feat, bm_feat):
        x = torch.cat([rm_feat, bm_feat], dim=1)
        x = self.fuse(x)
        return x


class PatchTF_RefineRM_BM_Stage2_Half_Skip6_No1x1(nn.Module):
    def __init__(self, heads, enc_layers, dec_layers, dim_ff, dropout):
        super().__init__()
        self.patch_size = 64
        self.grid = 4
        self.num_patches = 16
        self.d = 128

        if self.d % heads != 0:
            raise ValueError(f"d_model={self.d} must be divisible by heads={heads}")


        self.rm_enc = PatchEncoderPool5_Half_MS(in_ch=2)

        self.bm_enc = PatchEncoderPool5_Half_MS(in_ch=1)


        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.d))

        self.mod_embed = nn.Parameter(torch.zeros(2, self.d))


        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d,
            nhead=heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=enc_layers)


        dec_layer = nn.TransformerDecoderLayer(
            d_model=self.d,
            nhead=heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=dec_layers)


        self.rm_query = nn.Parameter(torch.zeros(1, self.num_patches, self.d))


        self.decode_fusion = DecodeConcatFusion_Half(in_ch=256, out_ch=128)


        self.skip_fuse8 = RM_BM_SkipFusion(rm_bm_in_ch=256, out_ch=128)
        self.skip_fuse16 = RM_BM_SkipFusion(rm_bm_in_ch=160, out_ch=80)
        self.skip_fuse32 = RM_BM_SkipFusion(rm_bm_in_ch=96, out_ch=48)
        self.skip_fuse64 = RM_BM_SkipFusion(rm_bm_in_ch=64, out_ch=32)
        self.skip_fuse128 = RM_BM_SkipFusion(rm_bm_in_ch=48, out_ch=24)
        self.skip_fuse256 = RM_BM_SkipFusion(rm_bm_in_ch=32, out_ch=16)


        self.global_decoder = GlobalDecoder128_From4_Skip6()


        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mod_embed, std=0.02)
        nn.init.trunc_normal_(self.rm_query, std=0.02)


    def patchify(self, x):

        B, C, H, W = x.shape
        p = self.patch_size
        g = self.grid
        patches = x.unfold(2, p, p).unfold(3, p, p)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(B, g * g, C, p, p)
        return patches

    def tokens_to_map(self, tok):

        B = tok.size(0)
        feat = tok.view(B, self.grid, self.grid, self.d)
        feat = feat.permute(0, 3, 1, 2).contiguous()
        return feat

    def patchfeat_to_wholemap(self, feat_patch, ch, h, w):

        B = feat_patch.size(0) // self.num_patches
        g = self.grid

        x = feat_patch.view(B, g, g, ch, h, w)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(B, ch, g * h, g * w)
        return x

    def forward(self, x_coarse, x_rm, x_bm, return_residual=False):

        B = x_coarse.size(0)
        p = self.patch_size


        x_rm2 = torch.cat([x_coarse, x_rm], dim=1)


        rm_p = self.patchify(x_rm2)
        bm_p = self.patchify(x_bm)


        rm_in = rm_p.view(B * self.num_patches, 2, p, p)
        bm_in = bm_p.view(B * self.num_patches, 1, p, p)

        rm_tok, rm_feats = self.rm_enc(rm_in)
        bm_tok, bm_feats = self.bm_enc(bm_in)

        rm_tok = rm_tok.view(B, self.num_patches, self.d)
        bm_tok = bm_tok.view(B, self.num_patches, self.d)


        x = torch.cat([rm_tok, bm_tok], dim=1)

        pos = torch.cat([self.pos_embed, self.pos_embed], dim=1)

        mod_rm = self.mod_embed[0].view(1, 1, self.d).expand(1, self.num_patches, self.d)
        mod_bm = self.mod_embed[1].view(1, 1, self.d).expand(1, self.num_patches, self.d)
        mod = torch.cat([mod_rm, mod_bm], dim=1)

        x = x + pos + mod


        memory = self.encoder(x)


        tgt = self.rm_query + self.pos_embed
        tgt = tgt.expand(B, self.num_patches, self.d)

        decoded_rm_tok = self.decoder(
            tgt=tgt,
            memory=memory
        )


        bm_mem_tok = memory[:, self.num_patches:self.num_patches * 2, :]


        rm_feat = self.tokens_to_map(decoded_rm_tok)
        bm_feat = self.tokens_to_map(bm_mem_tok)


        feat = self.decode_fusion(rm_feat, bm_feat)


        rm_skip8 = self.patchfeat_to_wholemap(rm_feats["s6"], ch=128, h=2, w=2)
        bm_skip8 = self.patchfeat_to_wholemap(bm_feats["s6"], ch=128, h=2, w=2)
        skip8 = self.skip_fuse8(rm_skip8, bm_skip8)

        rm_skip16 = self.patchfeat_to_wholemap(rm_feats["s5"], ch=80, h=4, w=4)
        bm_skip16 = self.patchfeat_to_wholemap(bm_feats["s5"], ch=80, h=4, w=4)
        skip16 = self.skip_fuse16(rm_skip16, bm_skip16)

        rm_skip32 = self.patchfeat_to_wholemap(rm_feats["s4"], ch=48, h=8, w=8)
        bm_skip32 = self.patchfeat_to_wholemap(bm_feats["s4"], ch=48, h=8, w=8)
        skip32 = self.skip_fuse32(rm_skip32, bm_skip32)

        rm_skip64 = self.patchfeat_to_wholemap(rm_feats["s3"], ch=32, h=16, w=16)
        bm_skip64 = self.patchfeat_to_wholemap(bm_feats["s3"], ch=32, h=16, w=16)
        skip64 = self.skip_fuse64(rm_skip64, bm_skip64)

        rm_skip128 = self.patchfeat_to_wholemap(rm_feats["s2"], ch=24, h=32, w=32)
        bm_skip128 = self.patchfeat_to_wholemap(bm_feats["s2"], ch=24, h=32, w=32)
        skip128 = self.skip_fuse128(rm_skip128, bm_skip128)

        rm_skip256 = self.patchfeat_to_wholemap(rm_feats["s1"], ch=16, h=64, w=64)
        bm_skip256 = self.patchfeat_to_wholemap(bm_feats["s1"], ch=16, h=64, w=64)
        skip256 = self.skip_fuse256(rm_skip256, bm_skip256)


        residual = self.global_decoder(
            feat,
            skip8=skip8,
            skip16=skip16,
            skip32=skip32,
            skip64=skip64,
            skip128=skip128,
            skip256=skip256,
        )


        final = x_coarse + residual


        bm_mask = (x_bm < 0.5).float().detach()
        final = final * bm_mask

        if return_residual:
            return final, residual
        return final


PatchTF_RM_BM_RefineStage2 = PatchTF_RefineRM_BM_Stage2_Half_Skip6_No1x1
