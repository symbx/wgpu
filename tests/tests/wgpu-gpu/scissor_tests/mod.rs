use wgpu_test::{apply, gpu_test, image, GpuTestConfiguration, GpuTestInitializer, TestingContext};

pub fn all_tests(vec: &mut Vec<GpuTestInitializer>) {
    vec.extend([
        SCISSOR_TEST_FULL_RECT,
        SCISSOR_TEST_EMPTY_RECT,
        SCISSOR_TEST_EMPTY_RECT_WITH_OFFSET,
        SCISSOR_TEST_CUSTOM_RECT,
        SCISSOR_TEST_MSAA_RESOLVE,
    ]);
}

struct Rect {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

const TEXTURE_HEIGHT: u32 = 2;
const TEXTURE_WIDTH: u32 = 2;
const BUFFER_SIZE: usize = (TEXTURE_WIDTH * TEXTURE_HEIGHT * 4) as usize;

async fn scissor_test_impl(
    ctx: &TestingContext,
    scissor_rect: Rect,
    sample_count: u32,
    expected_data: [u8; BUFFER_SIZE],
) {
    let texture_descriptor = wgpu::TextureDescriptor {
        label: Some("Offscreen texture"),
        size: wgpu::Extent3d {
            width: TEXTURE_WIDTH,
            height: TEXTURE_HEIGHT,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    };
    let texture = ctx.device.create_texture(&texture_descriptor);
    let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    // With multisampling, the pass renders into this texture and resolves into `texture`.
    let multisampled_view = (sample_count > 1).then(|| {
        ctx.device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("Multisampled texture"),
                sample_count,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                ..texture_descriptor
            })
            .create_view(&wgpu::TextureViewDescriptor::default())
    });

    let shader = ctx
        .device
        .create_shader_module(wgpu::include_wgsl!("solid_white.wgsl"));

    let pipeline = ctx
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Pipeline"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: sample_count,
                ..Default::default()
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

    let readback_buffer = image::ReadbackBuffers::new(&ctx.device, &texture);
    {
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        if multisampled_view.is_some() {
            // Stale contents, so a resolve that skips texels outside the scissor rect is visible.
            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Fill resolve target"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &texture_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::RED),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        }
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Renderpass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: multisampled_view.as_ref().unwrap_or(&texture_view),
                    depth_slice: None,
                    resolve_target: multisampled_view.as_ref().map(|_| &texture_view),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            render_pass.set_pipeline(&pipeline);
            render_pass.set_scissor_rect(
                scissor_rect.x,
                scissor_rect.y,
                scissor_rect.width,
                scissor_rect.height,
            );
            render_pass.draw(0..3, 0..1);
        }
        readback_buffer.copy_from(&ctx.device, &mut encoder, &texture);
        ctx.queue.submit(Some(encoder.finish()));
    }
    readback_buffer
        .assert_buffer_contents(ctx, &expected_data)
        .await;
}

#[apply(gpu_test!)]
static SCISSOR_TEST_FULL_RECT: GpuTestConfiguration =
    GpuTestConfiguration::new().run_async(|ctx| async move {
        scissor_test_impl(
            &ctx,
            Rect {
                x: 0,
                y: 0,
                width: TEXTURE_WIDTH,
                height: TEXTURE_HEIGHT,
            },
            1,
            [255; BUFFER_SIZE],
        )
        .await
    });

#[apply(gpu_test!)]
static SCISSOR_TEST_EMPTY_RECT: GpuTestConfiguration =
    GpuTestConfiguration::new().run_async(|ctx| async move {
        scissor_test_impl(
            &ctx,
            Rect {
                x: 0,
                y: 0,
                width: 0,
                height: 0,
            },
            1,
            [0; BUFFER_SIZE],
        )
        .await;
    });

#[apply(gpu_test!)]
static SCISSOR_TEST_EMPTY_RECT_WITH_OFFSET: GpuTestConfiguration = GpuTestConfiguration::new()
    .run_async(|ctx| async move {
        scissor_test_impl(
            &ctx,
            Rect {
                x: TEXTURE_WIDTH / 2,
                y: TEXTURE_HEIGHT / 2,
                width: 0,
                height: 0,
            },
            1,
            [0; BUFFER_SIZE],
        )
        .await
    });

#[apply(gpu_test!)]
static SCISSOR_TEST_CUSTOM_RECT: GpuTestConfiguration =
    GpuTestConfiguration::new().run_async(|ctx| async move {
        let mut expected_result = [0; BUFFER_SIZE];
        expected_result[((3 * BUFFER_SIZE) / 4)..][..BUFFER_SIZE / 4]
            .copy_from_slice(&[255; BUFFER_SIZE / 4]);

        scissor_test_impl(
            &ctx,
            Rect {
                x: TEXTURE_WIDTH / 2,
                y: TEXTURE_HEIGHT / 2,
                width: TEXTURE_WIDTH / 2,
                height: TEXTURE_HEIGHT / 2,
            },
            1,
            expected_result,
        )
        .await;
    });

#[apply(gpu_test!)]
static SCISSOR_TEST_MSAA_RESOLVE: GpuTestConfiguration =
    GpuTestConfiguration::new().run_async(|ctx| async move {
        let mut expected_result = [0; BUFFER_SIZE];
        expected_result[((3 * BUFFER_SIZE) / 4)..][..BUFFER_SIZE / 4]
            .copy_from_slice(&[255; BUFFER_SIZE / 4]);

        scissor_test_impl(
            &ctx,
            Rect {
                x: TEXTURE_WIDTH / 2,
                y: TEXTURE_HEIGHT / 2,
                width: TEXTURE_WIDTH / 2,
                height: TEXTURE_HEIGHT / 2,
            },
            4,
            expected_result,
        )
        .await;
    });
