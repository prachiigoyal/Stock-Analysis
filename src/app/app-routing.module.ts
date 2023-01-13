import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { PageAgreementComponent } from './pages/page-agreement/page-agreement.component';
import { PageBlogComponent } from './pages/page-blog/page-blog.component';
import { PageCategoryComponent } from './pages/page-category/page-category.component';
import { PageFaqsComponent } from './pages/page-faqs/page-faqs.component';
import { PageHomeComponent } from './pages/page-home/page-home.component';
import { PageNotFoundComponent } from './pages/page-not-found/page-not-found.component';
import { PagePrivacyComponent } from './pages/page-privacy/page-privacy.component';
import { PageProductComponent } from './pages/page-product/page-product.component';

const routes: Routes = [
  { path: '', component: PageHomeComponent },
  { path: 'blog', component: PageBlogComponent },
  { path: 'privacy', component: PagePrivacyComponent },
  { path: 'agreement', component: PageAgreementComponent },
  { path: 'faqs', component: PageFaqsComponent },

  {
    path: 'hair',
    children: [
      { path: '', component: PageCategoryComponent },
      { path: ':product', component: PageProductComponent },
    ],
  },
  {
    path: 'skin',
    children: [
      { path: '', component: PageCategoryComponent },
      { path: ':product', component: PageProductComponent },
    ],
  },
  {
    path: 'sex',
    children: [
      { path: '', component: PageCategoryComponent },
      { path: ':product', component: PageProductComponent },
    ],
  },
  { path: 'not-found', component: PageNotFoundComponent },
  { path: '**', redirectTo: '/not-found' },
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule],
})
export class AppRoutingModule {}
